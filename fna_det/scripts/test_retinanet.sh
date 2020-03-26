# !/usr/bin/env sh

python ./tools/test.py \
    ./configs/fna_retinanet_fpn_retrain.py \
    --checkpoint ./retinanet/retinanet.pth \
    --net_config ./retinanet/net_config \
    --data_path ./coco/ \
    --out ./results.pkl \
    --eval bbox \
