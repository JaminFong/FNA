# !/usr/bin/env sh

python -m torch.distributed.launch --nproc_per_node=8 ./tools/search.py \
            ./configs/fna_ssdlite_search.py \
            --launcher pytorch \
            --seed 1 \
            --work_dir ./ \
            --data_path ./coco/
