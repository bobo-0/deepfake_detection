# Deepfake Detection Project
- This repository contains all codes for Deepfake Detection Project.
- Team mates : Yebin Lee, Boyoung Han and Juhyeon Jung
- We did our project on Google colab with GPU and Server with GPU

## Main Idea
This project is aiming to investigate the effect of image change on Deepfake detection problem. Deepfake detection performance is compared using original (color) image, images with color changes, and images with changes in saturation.
You can find all the code we wrote in [here](/code). You can find example of data we used in [here](/data). There are some descriptions about our project below. You can read our [proposal report](/documents/DL_Project Proposal_Team1.pdf), [final report](/documents/Final_Report.pdf) and [final presentation](/documents/Final_PPT.pdf) to get more detail.. 

## Project
### Data
We used [Deepfake Detection Challenge dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) in Kaggle. To utilize the CNN model, we cut the video into frames and used it as images. The project was conducted with only 20% of the videos. We define the corresponding image as an Original Image (OI), an image with RGB values adjusted from the original image as an RGB Transformed Image (RTI), and finally an image with HSV values adjusted as an HSV Transformated Image (HTI).

### Data Preprocessing
1. Make different color type of video. 
    - [Original video to black and white video](/code/MakeGrayVideo.ipynb)
    - [Original video to high saturation video](/code/MakeHSVVideo.ipynb)
2. Split video to images.
    - [Split video as frame and save](/code/train_test_split.ipynb)

### Experiments
We did two big experiments with above data. In order to run model in better envrionment, we use server computer.
1. Find the best model with original image data
2. With the best model, run best data combination

#### 1. Find the best model
We used three type of pre-trained model : VGGFace2, ResNet50, DenseNet121.
- [VGGFace2](/code/vggface2.py)
- [ResNet50](/code/resnet50.py)
- [DenseNet121](/code/densenet121.py)

#### 2. Find the best data combination
Based on the result, we found that DenseNet121 has the best performance. 
Data combination:
1. Original Image(OI) only
2. RGB Transformed Image(RTI) only
3. HSV Transformed Image(HTI) only
4. OI + RTI
5. OI + HTI
6. OI + RTI + HTI

### Result
- The best performance model with our data : DensNet121
![Best Model](/documents/find_best.png)
- Among the models trained with one data group
    -  Highest performance is the model trained with HTI data
    -  Test accuracy recuded with RTI data
- Among the models trained with data combinations
    -  Performance increased because of data augmentation
    -  Data combinations with HTI data return HIGHEST performance  
![Model Result](/documents/densenet121.png)
