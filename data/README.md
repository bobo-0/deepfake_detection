# Example of Data
We used [Deepfake Detection Challenge dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) in Kaggle. To utilize the CNN model, we cut the video into frames and used it as images. The project was conducted with only 20% of the videos. We define the corresponding image as an Original Image (OI), an image with RGB values adjusted from the original image as an RGB Transformed Image (RTI), and finally an image with HSV values adjusted as an HSV Transformated Image (HTI).
You can download all data from kaggle and preprocess with the code we attached. There are some example of data below.
## Real data
<img src="https://github.com/bobo-0/deepfake_detection/blob/main/data/real_ori.jpg" width="400" height="300"/> <img src="https://github.com/bobo-0/deepfake_detection/blob/main/data/real_rti.jpg" width="400" height="300"/> <img src="https://github.com/bobo-0/deepfake_detection/blob/main/data/real_hti.jpg" width="400" height="300"/>


## Fake data
![Original Image](fake_ori.jpg) ![Original Image](fake_ori_2.jpg)
![RGB Transformed Image](fake_rti.jpg) ![RGB Transfomred Image](fake_rti_2.jpg)
![HSV Transformed Image](fake_hti.jpg) ![HSV Transformed Image](fake_hti_2.jpg)
