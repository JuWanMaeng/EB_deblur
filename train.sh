python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train_EB.py -opt options/train/NAFNet/EB_NAFNet.yml --launcher pytorch