import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import os, wandb

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class ImageReconRestorationModel(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageReconRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        
        self.NAFVAE = define_network(deepcopy(opt['network_NAFVAE']))
        self.NAFVAE = self.model_to_device(self.NAFVAE)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            
        load_path_NAFVAE = self.opt['path'].get('pretrain_network_NAFVAE', None)
        if load_path_NAFVAE is not None:
            self.load_network(self.NAFVAE, load_path_NAFVAE,
                              self.opt['path'].get('strict_load_NAFVAE', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        if 'train' in opt['datasets']:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.init(project='promptir')
                wandb.run.name = opt['name']
            self.wandb = True
        else:
            self.wandb = False

    def init_training_settings(self):
        self.net_g.train()
        self.NAFVAE.train()

        # 2) freeze 할 encoder 모듈들
        encoder_modules = [
            self.NAFVAE.intro,
            *self.NAFVAE.encoders,     # encoder stage
            *self.NAFVAE.downs,        # 다운샘플링
            self.NAFVAE.middle_encoder,
            self.NAFVAE.latent_to_8,   # latent 변환 앞부분
        ]
        for m in encoder_modules:
            for p in m.parameters():
                p.requires_grad = False

        # 3) trainable 할 decoder 모듈들
        decoder_modules = [
            self.NAFVAE.latent_from_8,
            self.NAFVAE.middle_decoder,
            *self.NAFVAE.ups,
            *self.NAFVAE.decoders,
            self.NAFVAE.ending,
        ]
        for m in decoder_modules:
            for p in m.parameters():
                p.requires_grad = True

        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None


        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        from copy import deepcopy
        train_opt = self.opt['train']
        optim_cfg = deepcopy(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type')
        base_lr = optim_cfg.pop('lr')

        net_g_params = []
        nafvae_params = []
        logger = get_root_logger()

        # 1) Collect trainable parameters from net_g
        for name, param in self.net_g.named_parameters():
            if param.requires_grad:
                net_g_params.append(param)
            else:
                logger.warning(f'net_g parameter "{name}" frozen, will not be optimized.')

        # 2) Collect trainable parameters from NAFVAE (decoder-only)
        for name, param in self.NAFVAE.named_parameters():
            if param.requires_grad:
                nafvae_params.append(param)
            else:
                logger.warning(f'NAFVAE parameter "{name}" frozen, will not be optimized.')

        # 3) Create two param groups: 
        #    - net_g at base_lr
        #    - NAFVAE decoder at base_lr / 10
        param_groups = [
            {'params': net_g_params,    'lr': base_lr},
            {'params': nafvae_params,   'lr': base_lr * 0.1}
        ]

        # 4) Instantiate optimizer
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(param_groups, **optim_cfg)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(param_groups, **optim_cfg)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {optim_type}")

        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.lq = data['frame'].to(self.device)
        self.lq = self.lq.float()

        if 'frame_gt' in data:
            self.gt = data['frame_gt'].to(self.device)

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()


        preds = self.net_g(self.lq)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.

            if self.pixel_type == 'PSNRATLoss':
                l_pix += self.cri_pix(*preds, self.gt)

            elif self.pixel_type == 'PSNRLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)
            
            else:
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)    

            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # fft loss
        if self.cri_fft:
            l_fft = self.cri_fft(preds[-1], self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft         

        


        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if current_iter % 10 ==0:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.log({'train_loss': loss_dict['l_pix'].item(), 'iter':current_iter})

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                pred = self.net_g(self.lq[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()
            # self.grids_inverse_voxel()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name') # !
        
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):


            img_name = '{:06d}'.format(cnt)


            lq = val_data['frame']
            event = val_data['gen_event']


            lq = torch.cat([lq,event],dim=(1))
            val_data['frame'] = lq
            
            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    if cnt == 1: # visualize cnt=1 image every time
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}.png')
                        
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    print('Save path:{}'.format(self.opt['path']['visualization']))
                    print('Dataset name:{}'.format(dataset_name))
                    print('Img_name:{}'.format(img_name))
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

    
            # default setting
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)



            pbar.update(1)
            # pbar.set_description(f'Test {img_name}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'

        if self.wandb:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.log({'val_loss': value, 'iter':current_iter})
            
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
