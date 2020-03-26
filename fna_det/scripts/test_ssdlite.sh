# !/usr/bin/env sh

python ./tools/test.py \
    ./configs/fna_ssdlite_retrain.py \
    --checkpoint ./ssdlite/ssdlite.pth \
    --net_config ./ssdlite/net_config \
    --data_path ./coco/ \
    --out ./results.pkl \
    --eval bbox \
