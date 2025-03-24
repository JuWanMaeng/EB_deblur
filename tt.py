import wandb

wandb.init(project='promptir')
wandb.run.name = 'EFNet'

iters = [1000]
for i in range(1,21):
    num = i*10000
    iters.append(num)
# scores = [30.1696,32.684, 33.9449, 34.4943,34.6069,34.6172, 35.0420, 35.0462,  35.1496,35.1683, 35.1407, 35.4019,
#           35.3688, 35.3236,35.4590,35.4606, 35.6180,  35.6048,  35.5643,35.5813, 35.5996, 35.6031, 35.5377,
#            35.6306,  ]

scores = [35.46 for _ in range(len(iters))]

for i in range(len(scores)):
    wandb.log({'val_loss': scores[i], 'iter':iters[i]})