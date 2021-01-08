# real_time_FER_using_DL

Link of paper: http://jdatasci.com/index.php/jdatasci/article/view/4

# Abstract
In this study, we detected the faces on real-time video data to recognize the anger, fear, happy, surprise, sad and neutral emotions upon these detected faces using deep learning methods. We created our own dataset to use in this study for six different facial emotions. At first stage, we created a convolutional neural network and trained it over our dataset by scratching method and we achieved 50% accuracy rate. Then, we increased the number of images in our database by 3 times, and get better accuracy which is 62%. Thanks to transfer training method and AlexNet's pre-trained networks, we reached 74% accuracy rate after increasing the number of images 80% in the dataset. In addition, we achieved 72% accuracy rate when we test our network which is trained with our own dataset with the Compound Emotion dataset. The basic reason of this decrease can be angry emotion because there are differences poses between our dataset and Compound Emotion dataset for angry emotion images. However, we obtained 100% accuracy rate for happy emotion and 89% for sad emotion. It has been seen that the work we are doing gives successful results when tested with different people in different ambient and light conditions.

# Requirements

* MATLAB R2018b
* Deep Learning Toolbox


# Citation

Please cite as:

E. Dandıl and R. Özdemir, “Real-time Facial Emotion Classification Using Deep Learning”, DataSCI, vol. 2, no. 1, pp. 13-17, Jul. 2019.
