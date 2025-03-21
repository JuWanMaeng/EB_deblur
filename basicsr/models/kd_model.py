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


class KnowledgeDistillationModel(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(KnowledgeDistillationModel, self).__init__(opt)

        # teacher model
        self.net_t = define_network(deepcopy(opt['network_teacher']))
        self.net_t = self.model_to_device(self.net_t)

        t_load_path = self.opt['t_path'].get('pretrain_network', None)
        if t_load_path is not None:
            self.load_network(self.net_t, t_load_path,
                              self.opt['t_path'].get('strict_load_g', True), param_key=self.opt['t_path'].get('param_key', 'params'))
            self.net_t.eval()
            print('Teacher model loaded complete')

        # student model
        self.net_s = define_network(deepcopy(opt['network_student']))
        self.net_s = self.model_to_device(self.net_s)

        s_load_path = self.opt['s_path'].get('pretrain_network', None)
        if s_load_path is not None:
            self.load_network(self.net_s, s_load_path,
                              self.opt['s_path'].get('strict_load_g', True), param_key=self.opt['s_path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        if 'train' in opt['datasets']:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.init(project='Knowledge Distillation')
                wandb.run.name = 'NAFNet'
            self.wandb = True
        else:
            self.wandb = False

        self.lambda_pix, self.lambda_enc, self.lambda_dec, self.lambda_mid = 1, 0.5, 0.5, 0.25

    def init_training_settings(self):
        self.net_s.train()
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


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_s.named_parameters():
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

        optim_type = train_opt['optim_s'].pop('type')

        if optim_type == 'Adam':
            self.optimizer_s = torch.optim.Adam(
                [{'params': optim_params},
                {'params': optim_params_lowlr, 'lr': train_opt['optim_s']['lr'] * ratio}],
                **train_opt['optim_s']
            )
        elif optim_type == 'AdamW':
            self.optimizer_s = torch.optim.AdamW(
                                                    [{'params': optim_params},
                {'params': optim_params_lowlr, 'lr': train_opt['optim_s']['lr'] * ratio}],
                **train_opt['optim_s']
            )
        else:
            raise NotImplementedError(f"Optimizer {optim_type} is not implemented")

        self.optimizers.append(self.optimizer_s)

    def feed_data(self, data):

        self.lq = data['frame'].to(self.device)
        self.gen_event = data['gen_event'].to(self.device)
        self.original_voxel = data['original_voxel'].to(self.device)
        self.refined_event = data['refined_event'].to(self.device)

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
        self.optimizer_s.zero_grad()

        teacher_input = torch.cat([self.lq,self.original_voxel], dim=1)
        student_input = torch.cat([self.lq, self.gen_event], dim=1)

        with torch.no_grad():
            teacher_outputs = self.net_t(teacher_input)
            # teacher_outputs: (teacher_out, teacher_enc_feats, teacher_dec_feats, teacher_mid_first, teacher_mid_last)
            teacher_out, teacher_enc_feats, teacher_dec_feats, teacher_mid_first, teacher_mid_last = teacher_outputs

        # Student 모델 forward
        student_outputs = self.net_s(student_input)
        # student_outputs: (student_out, student_enc_feats, student_dec_feats, student_mid_first, student_mid_last)
        student_out, student_enc_feats, student_dec_feats, student_mid_first, student_mid_last = student_outputs


        self.output = student_out  # 최종 student 출력 저장

        l_total = 0
        loss_dict = OrderedDict()

        # 1. Pixel Loss (예: MSE, PSNRLoss 등)
        if self.cri_pix:
            l_pix = self.cri_pix(student_out, teacher_out)
            loss_dict['l_pix'] = l_pix.item()
            l_total += l_pix * self.lambda_pix


        # 2. Encoder Feature Loss (모든 encoder 단계 feature 평균)
        l_enc = 0
        for s_feat, t_feat in zip(student_enc_feats, teacher_enc_feats):
            l_enc += F.mse_loss(s_feat, t_feat)
        if len(student_enc_feats) > 0:
            l_enc = l_enc / len(student_enc_feats)
        loss_dict['l_enc'] = l_enc.item()
        l_total += l_enc * self.lambda_enc
     
        # 3. Decoder Feature Loss (모든 decoder 단계 feature 평균)
        l_dec = 0
        for s_feat, t_feat in zip(student_dec_feats, teacher_dec_feats):
            l_dec += F.mse_loss(s_feat, t_feat)
        if len(student_dec_feats) > 0:
            l_dec = l_dec / len(student_dec_feats)
        loss_dict['l_dec'] = l_dec.item()
        l_total += l_dec * self.lambda_dec

        # 4. Middle Feature Loss (middle 블록 중 첫 번째와 마지막 feature 사용)
        l_mid_first = F.mse_loss(student_mid_first, teacher_mid_first)
        l_mid_last = F.mse_loss(student_mid_last, teacher_mid_last)
        l_mid = (l_mid_first + l_mid_last) / 2
        loss_dict['l_mid'] = l_mid.item()
        l_total += l_mid * self.lambda_mid

        loss_dict['l_total'] = l_total.item()
        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_s.parameters(), 0.01)
        self.optimizer_s.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if current_iter % 10 ==0:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                # wandb.log({'train_loss': l_total.item(), 'iter':current_iter})
                wandb.log({'output_loss': loss_dict['l_pix'].item(), 'iter':current_iter})
                wandb.log({'enc_loss': loss_dict['l_enc'].item(), 'iter':current_iter})
                wandb.log({'dec_loss': loss_dict['l_dec'].item(), 'iter':current_iter})
                wandb.log({'mid_loss': loss_dict['l_mid'].item(), 'iter':current_iter})
                wandb.log({'total_loss': loss_dict['l_total'].item(), 'iter':current_iter})

    def test(self):
        self.net_s.eval()
        with torch.no_grad():
            n = self.lq.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                student_input = torch.cat([self.lq, self.gen_event], dim=1)
                student_out = self.net_s(student_input[i:j, :, :, :])[0]  # mini batch all in 
            
                if isinstance(student_out, list):
                    student_out = student_out[-1]
                outs.append(student_out)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_s.train()

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
        out_dict['gt'] = self.refined_event.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_s, 'net_s', current_iter)
        self.save_training_state(epoch, current_iter)
