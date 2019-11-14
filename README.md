# HomeProTech
## Tech to Protect Contest 5 Hackathon Submission:  Fire Safety in 3D:  Incentivizing Homeowners to Create Pre-Incident Plan for Firefighters

![teaser](https://github.com/sshaoshuai/PointRCNN/blob/master/doc/teaser.png)

Code release for the Tech to Protect Online National Hackathon, November 2019.

**Authors**:Emily, Katherine, Judson, Stephanie.

[[arXiv]](https://arxiv.org/abs/1812.04244)&nbsp;  [[Project Page]](#)&nbsp;

## Introduction
In this work, we propose the

For more details , please refer to [our proto-type](https://arxiv.org/abs/1812.04244) or [project page](#).


## Installation
### Requirements
All the codes are tested in the following environment:

* Python 3.6+
* PyTorch 1.0

### To Use
https://github.com/StephanieRogers-ML/HomeProTech/edit/master/README.md
a. Clone the HomeProTech repository.
```shell
git clone --recursive https://github.com/StephanieRogers-ML/HomeProTech.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
HomeProTech
├── data
│   ├── vizHome
│   │   ├── XYZ-RGB File
│   │   ├── Txt File
│   │   ├──training
│   │   ├──testing
│   ├── vizHome
├── lib
├── pointnet2_lib
├── tools
```
Here the images are only used for visualization and the [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) are optional for data augmentation in the training. 


## Pretrained model
You could download the pretrained model(ShapeNet)  from [here(~15MB)], which is trained on the *train* split (3712 samples) and evaluated on the *val* split (3769 samples) and *test* split (7518 samples). The performance on validation set is as follows:
```
Shape AP@0.70, 0.70, 0.70:
bbox AP:96.91, 89.53, 88.74
bev  AP:90.21, 87.89, 85.51
3d   AP:89.19, 78.85, 77.91
aos  AP:96.90, 89.41, 88.54
```
### Quick demo
You could run the following command to evaluate the pretrained model: 
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt PointRCNN.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False
```

## Inference
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rcnn 
```

* To evaluate all the checkpoints of a specific training config file, add the `--eval_all` argument, and run the command as follows:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --eval_mode rcnn --eval_all
```

* To generate the results on the *test* split, please modify the `TEST.SPLIT=TEST` and add the `--test` argument. 

Here you could specify a bigger `--batch_size` for faster inference based on your GPU memory. Note that the `--eval_mode` argument should be consistent with the `--train_mode` used in the training process. If you are using `--eval_mode=rcnn_offline`, then you should use `--rcnn_eval_roi_dir` and `--rcnn_eval_feature_dir` to specify the saved features and proposals of the validation set. Please refer to the training section for more details. 

