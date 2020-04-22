# HomeProTech

https://home-pro-tec.herokuapp.com/
## Tech to Protect Contest 5 Hackathon Submission:  Fire Safety in 3D:  Incentivizing Homeowners to Create Pre-Incident Plan for Firefighters

Code release for the Tech to Protect Online National Hackathon, November 2019.
Online Contest Submission, January 2020.  

https://www.techtoprotectchallenge.org/from-interior-design-to-ux-at-the-denver-tech-to-protect-challenge-hackathon/  


**Authors**:Emily, Katherine, Judson, Stephanie.
https://github.com/kalex19/Home-Pro-Tech

![Future](https://github.com/StephanieRogers-ML/HomeProTech/blob/master/pytorch-flask-api-heroku-home-pro-tec/static/ModelFlow.jpg)
# Tech To Protect - Contest #5 

## Home Pro-Tech
### Introduction
This project was started to compete in the [Tech To Protect](https://www.techtoprotectchallenge.org/) challenge, specificially for [contest #5](https://www.techtoprotectchallenge.org/contest/contest-005/). We created the idea in order to address what we saw as a large gap in the use of technology by firefighters and first responders. While pursuing our idea, we realized that there is also a viable business idea in using 3D modeling and machine learning in order to play the middle man between home insurance companies, city and county fire departments, and homeowners. Using the data generated by the user and the automated generation of the machine learning models, we will be able to increase the efficiency of the emergency responses to houses within our network, while also providing savings to the homeowners by ensuring that their homeowners insurance company is continually updated with their pre-incident plan and fire safety checklist. 

### Knowledge Gathering
We interviewed multiple firefighters in the Denver area, and were able to compile a large amount of information concerning the current gaps in emergency responses. These included not having any idea of home floor plans, not knowing if residents of the home had medical conditions that could interfere with rescues, and not having any idea of the clutter level or hazards that could be in the home. These problems could be solved using machine learning on point clouds in order to do item detection and floor plan generation.

### Process
This will be accomplished by incentivising the homeowners to take multiple pictures and/or videos of each room and using those photos or videos to create a point cloud rendering of the room. This will allow the creation of floor plans using this data and also of image recognition using the images/videos and the point cloud. This would allow us to detect various things about the evironment, included the presence of safety equipment and verification that the homeowner is following the commonly accepted safety regulations for fire protection.

This data can then be sent to the homeowners insurance company along with the local fire department. The insurance company could give the homeowner a premium break for being in compliance with security practices. The local fire department could access the information about the homeowners house in order to make any emergency response much more efficient. The most requested feature from the firefighters we interviewed was information about the residents of the house. We would prompt the homeowner to upload that information, including non-specific health conditions that would allow the firefighters to plan for residents in a wheelchar, for example. Using the 3D models generated from the homeowners pictures and videos, we would also generate a 3D model of the house for the firefighter to explore as they respond to the emergency. 

### Technology
Currently we are iterating through the different aspects of this application. We envision a React Native mobile application, utilizing a Flask back-end with a Postgres database, running on Amazon Web Services (AWS) or Microsoft Azure. This would allow the secure storage of homeowner information, and allow the mobile application to communicate securely with the back-end. The Flask back-end will encompass a couple different machine learning models. One of those models will process the stitched together point cloud images in order to recognize items inside the rooms. Another model will take the pictures and video and provide item detection using computer vision. The third model will perform analysis of the image outputs of the cloud slicing script, which will slice the point cloud file into slices that are then turned into JPEGs. This analysis will determine the floor plan of the house.

The image analysis will also be used to help pre-populate a fire safety checklist. Utilizing a final machine learning model, we will use Q learning to attempt to compute the most efficient escape routes from each room, utilizing the information we have about the floor plan and rest of the house in order to navigate to floor level windows and doors. All of this information will be verified by the homeowner before being finalized.

## Presentation Information
You can find a Canva presentation at [this link](https://www.canva.com/design/DADqzVUakYU/tXl77_Pf179MxeIRib-KvQ/view?utm_content=DADqzVUakYU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink). You can find the three different prototypes at these links - [the sign up process for new users](https://invis.io/8JUVC43CMAG), [the homeowner experience](https://invis.io/CRUVC5ANPB7), and [the firefighter experience](https://invis.io/T8UVBN1MSWK).

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



