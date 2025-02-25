import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import os, wandb
import numpy as np

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class ImageEventRefinementModel(BaseModel):
    def __init__(self, opt):
        super(ImageEventRefinementModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        if 'train' in opt['datasets']:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.init(project='promptir')
                wandb.run.name = '(NAF)Event refinement'
            self.wandb = True
        else:
            self.wandb = False

    def init_training_settings(self):
        self.net_g.train()
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

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('wae_opt'):
            wae_loss_type = train_opt['wae_opt'].pop('type')
            cri_wae_cls = getattr(loss_module, wae_loss_type)
            self.cri_wae = cri_wae_cls(**train_opt['wae_opt']).to(self.device)
        else:
            self.cri_wae = None

        if train_opt.get('fft_loss_opt'):
            fft_loss_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_loss_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(self.device)
        else:
            self.cri_fft = None



        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.gen_event = data['gen_event'].to(self.device) # input
        self.lq = data['frame'].to(self.device)

        self.voxel=data['voxel'].to(self.device) # GT

        self.input = torch.cat([self.gen_event, self.lq], dim=1)


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
        preds = self.net_g(self.input)


        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.

            if self.pixel_type == 'PSNRLoss':
                for pred in preds:
                    l_pix += self.cri_pix(self.output, self.voxel)
            
            else:
                for pred in preds:
                    l_pix += self.cri_pix(self.output, self.voxel)    

            l_total += l_pix
            loss_dict['l_pix'] = l_pix


        # FFT loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.voxel)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft 


        # KL loss
        if self.cri_wae:
            l_kl = 0
            l_kl += self.cri_wae(self.output,self.voxel)
            l_total += l_kl
            loss_dict['l_wae'] = l_kl



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
                if self.cri_pix:
                    wandb.log({'train_loss': loss_dict['l_pix'].item() / self.cri_pix.loss_weight, 'iter':current_iter})
                if self.cri_wae:
                    wandb.log({'wae_loss': loss_dict['l_wae'].item() / self.cri_wae.loss_weight, 'iter':current_iter})
                if self.cri_fft:
                    wandb.log({'fft_loss': loss_dict['l_fft'].item() / self.cri_fft.loss_weight, 'iter':current_iter})

                wandb.log({'total_loss': l_total.item() , 'iter':current_iter})


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.input.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                    b, c, h, w = self.input[i:j].shape
                    h_n = (32 - h % 32) % 32
                    w_n = (32 - w % 32) % 32
                    in_tensor = F.pad(self.input[i:j], (0, w_n, 0, h_n), mode='reflect')
                    self.input = in_tensor

                    pred = self.net_g(self.input)
                    # pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                pred = pred[:, :, :h, :w]
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

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()


            ###############
            # mask = (torch.abs(visuals['gt']) <= 0.05).bool()
            # visuals['gt'].masked_fill_(mask, 0)
            ###############


            # tentative for out of GPU memory
            del self.input
            del self.lq
            del self.gen_event
            del self.output
            torch.cuda.empty_cache()


            # default setting
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])

                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                    
            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'],
                                                val_data['path'][0])
                os.makedirs(save_img_path, exist_ok=True)
                output_event = visuals['result'][0].numpy()
                np.save(f'{save_img_path}/out.npy', output_event)
            

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
        out_dict['lq'] = self.gen_event.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'voxel'):
            out_dict['gt'] = self.voxel.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)