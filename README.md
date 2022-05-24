# Deepfake Detection Project
- This repository contains all codes for Deepfake Detection Project.
- Team mates : Yebin Lee, Juhyeon Jung
- We did our project on Google colab with GPU. and Server with GPU

## Main Idea
This project is aiming to investigate the effect of image change on Deepfake detection problem. Deepfake detection performance is compared using original (color) image, images with color changes, and images with changes in saturation.
You can read our proposal document in [here](documents/). You can find all the code we wrote in [here](/code).

## Project
### Data
We used [Deepfake Detection Challenge dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) in Kaggle. 

### Data Preprocessing
We did data preprocessing to make different type of video. 
- [Original video to black and white video](/code/MakeGrayVideo.ipynb)
- [Original video to high saturation video](/code/MakeHSVVideo.ipynb)
