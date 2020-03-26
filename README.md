# FNA

The code of the ICLR 2020 paper [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://openreview.net/forum?id=rklTmyBKPH)

Deep neural networks achieve remarkable performance in many computer vision tasks. Most state-of-the-art (SOTA) semantic segmentation and object detection approaches reuse neural network architectures designed for image classification as the backbone, commonly pre-trained on ImageNet. However, performance gains can be achieved by designing network architectures specifically for detection and segmentation, as shown by recent neural architecture search (NAS) research for detection and segmentation. One major challenge though, is that ImageNet pre-training of the search space representation (a.k.a. super network) or the searched networks incurs huge computational cost. 

In this paper, we propose a Fast Neural Network Adaptation (FNA) method, which can adapt both the architecture and parameters of a seed network (e.g. a high performing manually designed backbone) to become a network with different depth, width, or kernels via a Parameter Remapping technique, making it possible to utilize NAS for detection/segmentation tasks a lot more efficiently.

![framework](./imgs/framework.png)

## Results

In our experiments, we conduct FNA on MobileNetV2 to obtain new networks for both segmentation and detection that clearly out-perform existing networks designed both manually and by NAS. The total computation cost of FNA is significantly less than SOTA segmentation/detection NAS approaches: 1737x less than DPC, 6.8x less than Auto-DeepLab and 7.4x less than DetNAS.

<div  align="center">
<img src="./imgs/seg_results.png" width = "700">
<img src="./imgs/seg_cost.png" width = "600">
<img src="./imgs/det_results.png" width = "550">
<img src="./imgs/det_cost.png" width = "700">
</div>

## FNA on Object Detection

### Requirements

* python 3.7
* pytorch 1.1 
* mmdet 0.6.0 (53c647e)
* mmcv 0.2.10

### Architecture Adaptation

Adapt the architecture of the seed network to the target dataset COCO. The adaptation process is performed on 8 TITAN-XP GPUs. First go to the path of the detection project `cd fna_det`.
* RetinaNet: `sh scripts/arch_adapt_retinanet.sh`
* SSDLite: `sh scripts/arch_adapt_ssdlite.sh`

### Parameter Adaptation

Adapt the parameters of the target architecture on COCO. The adaptation is performed on 8 TITAN-XP GPUs.

```bash
cd fna_det
```
* RetinaNet: `sh scripts/param_adapt_retinanet.sh`
* SSDLite: `sh scripts/param_adapt_ssdlite.sh`

The seed network MobileNetV2 is trained on ImageNet using the code of [DenseNAS](https://github.com/JaminFong/DenseNAS). We provide the pre-trained weights and `net_config` of the seed network in [MobileNetV2](https://drive.google.com/open?id=1XW0NxkLckKQ68s6V7nf7vF4qe1WsL3GE). The code of MobileNetV2 model is constructed in `models/derived_imagenet_net.py`.

### Evaluation

We provide the adapted parameters and `net_config` in checkpoint [RetinaNet](https://drive.google.com/open?id=1BatmfFQ6ArcYN3l8OD9epl3asRhjx1yx) and [SSDLite](https://drive.google.com/open?id=1iIzlctJj8VJgsCVlXEe6MVoq3SEza964). The complete model zoo is in [FNA_modelzoo](https://drive.google.com/open?id=10iH62XcE5AVGDXEa-yjeaAjK6isUh1QD). You can evaluate the checkpoint with the following script.

* RetinaNet: `sh scripts/test_retinanet.sh`
* SSDLite: `sh scripts/test_ssdlite.sh`

## FNA on Semantic Segmentation

### Requirements

* python 3.7
* pytorch 1.1

### Evaluation

1. The adapted parameters and `net_config` are available in [DeepLabV3](https://drive.google.com/open?id=1dNK5QEn-Mhcx20fctz7Zo9yZzDzCvyBt).

2. Put the adapted parameters `epoch-last.pth` into `fna_seg/model/deeplab/cityscapes.deeplabv3`.

3. You can evaluate the checkpoint with the following script. 

    ```bash
    cd fna_seg/model/deeplab/cityscapes.deeplabv3
    sh eval.sh.
    ```

## Citation

```bash
@inproceedings{
    fang2020fast,
    title={Fast Neural Network Adaptation via Parameter Remapping and Architecture Search},
    author={Jiemin Fang* and Yuzhu Sun* and Kangjian Peng* and Qian Zhang and Yuan Li and Wenyu Liu and Xinggang Wang},
    booktitle={International Conference on Learning Representations},
    year={2020},
}
```

## Acknowledgement

The code of FNA is based on 

* [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v0.6.0)
* [TorchSeg](https://github.com/ycszen/TorchSeg)

Thanks for the contribution of the above repositories.
