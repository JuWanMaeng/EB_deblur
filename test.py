import wandb

wandb.init(project='promptir')
wandb.run.name = 'NAFNet'

iters = [i for i in range(0,300000,50000)]
psnr = [33.69 for _ in range(6)]

for i in range(len(iters)):
    wandb.log({'val_loss': psnr[i], 'iter':iters[i]})