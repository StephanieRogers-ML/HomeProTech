# HomeProTech
## Tech to Protect Contest 5 Hackathon Submission:  Fire Safety in 3D:  Incentivizing Homeowners to Create Pre-Incident Plan for Firefighters

Code release for the Tech to Protect Online National Hackathon, November 2019.

**Authors**:Emily, Katherine, Judson, Stephanie.
https://github.com/kalex19/Home-Pro-Tech


## Installation
### Requirements
All the codes are tested in the following environment:

* Python 3.6+
* PyTorch 1.0

### To Use 
## PointNet trained on modelnet40 & shapenet with Pytorch
```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```
https://github.com/StephanieRogers-ML/HomeProTech/edit/master/README.md
a. Clone the HomeProTech repository.
```shell
git clone --recursive https://github.com/StephanieRogers-ML/HomeProTech.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.


## Dataset preparation
Part Segmentation - ShapeNetPart dataset
The main datasets used for transfer learning for point cloud classification and segmentation are: 
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


