CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=4321 hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml --launcher pytorch