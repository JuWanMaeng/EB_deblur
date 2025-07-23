# python setup.py develop --no_cuda_ext
torchrun --nproc_per_node=2 --master_port=5431 basicsr/train_EB.py \
    -opt /workspace/FFTformer/options/train/Restormer/Restormer.yml \
    --launcher pytorch