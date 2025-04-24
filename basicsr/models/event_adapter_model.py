import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os, wandb
import numpy as np

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from diffusers import AutoencoderKL

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')



class EventAdapterModel(BaseModel):
    """Base Event-based deblur model with Domain Adapter training."""

    def __init__(self, opt):
        super().__init__(opt)

        self.rgb_latent_scale_factor = 0.18215
        # define main VAE network (event VAE)
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)


        # load pretrained VAE weights (excluding adapter)
        pretrain_path = opt['path'].get('pretrain_network_g', None)
        if pretrain_path:
            ckpt = torch.load(pretrain_path, map_location=self.device)
            # drop adapter.* keys if present
            state = {k: v for k, v in ckpt['params'].items() if not k.startswith('adapter.')}
            self.net_g.load_state_dict(state, strict=False)

        # load SDv2 VAE for target latent
        self.sd_vae = AutoencoderKL.from_pretrained(opt['path']['sd_vae_path'])
        self.sd_vae = self.sd_vae.to(self.device).eval()
        for p in self.sd_vae.parameters(): p.requires_grad = False

        if self.is_train:
            self.init_training_settings()

        if 'train' in opt['datasets']:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.init(project='Adapter')
                wandb.run.name = self.opt['name']
            self.wandb = True
        else:
            self.wandb = False

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        # define pixel loss if any
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None
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

        ratio = 0.1
        optim_type = train_opt['optim_g'].pop('type')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                [{'params': optim_params},
                {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                **train_opt['optim_g']
            )
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                                                    [{'params': optim_params},
                {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                **train_opt['optim_g']
            )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.event = data['voxel'].to(self.device)


    def optimize_parameters(self, current_iter):
        # zero grad
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        # forward through event VAE to get latent
        z_evt = self.net_g.encode(self.event)
        # map to SD latent
        z_pred = self.net_g.map_to_sd_latent(z_evt)
        # get target SD latent
        with torch.no_grad():
            event_latent = self.encode_sd_event(self.event)

        l_total = 0
        loss_dict = OrderedDict()
        
        # 1. Pixel loss 계산
        if self.cri_pix:
            l_pix = 0.
            l_pix += self.cri_pix(z_pred, event_latent)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # backward and step
        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if current_iter % 10 == 0:
            local_rank = os.environ.get('LOCAL_RANK', '0')
            if local_rank == '0':
                wandb.log({
                    'step_loss': loss_dict['l_pix'].item(),
                    # 'kl_loss': loss_dict['l_kl'].item()/ self.cri_kl_weight,
                    # 'total_loss': l_total.item(),
                    'iter': current_iter
                })

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.event.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                # pred,_,_ = self.net_g(self.event[i:j, :, :, :])  # mini batch all in 
                pred = self.net_g(self.event[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        self.test()
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
            sr_img = visuals['result']
            gt_img = visuals['gt']


            # tentative for out of GPU memory
            del self.event
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'],
                                                val_data['path'][0])
                os.makedirs(save_img_path, exist_ok=True)
                output_event = visuals['result'][0].numpy()
                np.save(f'{save_img_path}/out.npy', output_event)

    
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
        out_dict['result'] = self.output.detach().cpu()
        out_dict['gt'] = self.event.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


    def encode_sd_event(self, six_event):

        event_1 = six_event[:, 0:3, :, :]
        event_2 = six_event[:, 3:, :, :]

        # encode
        h_1 = self.sd_vae.encoder(event_1)
        moments_1 = self.sd_vae.quant_conv(h_1)
        mean_1, logvar = torch.chunk(moments_1, 2, dim=1)
        # scale latent
        event_latent_1 = mean_1 * self.rgb_latent_scale_factor

        h_2 = self.sd_vae.encoder(event_2)
        moments_2 = self.sd_vae.quant_conv(h_2)
        mean_2, logvar = torch.chunk(moments_2, 2, dim=1)
        # scale latent
        event_latent_2 = mean_2 * self.rgb_latent_scale_factor

        event_latent = torch.cat([event_latent_1,event_latent_2], dim = 1)

        return event_latent
    