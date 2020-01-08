# !/usr/bin/env sh

# python -m torch.distributed.launch --nproc_per_node=8 ./tools/retrain.py \
#             ./configs/fna_retinanet_fpn_retrain.py \
#             --launcher pytorch \
#             --seed 1 \
#             --work_dir ./ \
#             --data_path ./coco/ \
#             --validate \

python -m torch.distributed.launch --nproc_per_node=1 ./tools/retrain.py \
            ./configs/fna_ssdlite_retrain.py \
            --launcher pytorch \
            --seed 1 \
            --work_dir ./ \
            --data_path /mnt/data-3/data/jiemin.fang/dataset/coco/ \
            --validate \