## Sports-classifier-using-ResNet-50 (95% accuracy)

This repository contains a deep learning project where a model was built using the base layers of pre-trained ResNet-50 model. The model was trained on 13,000 images of sports images and achieved an accuracy of 95% on the test set.
To futher improve the model's accuracy, 65 top layers of the base model (ResNet-50) were set to trainable, and trained using a learning rate lower by a factor of 10 than the initial learning rate.

**Overall we were not able to achieve a higher accuracy, as the model had an accuracy of 94.4% compared to the initial 95.2% accuracy. However, this technique can sometimes increase the performance of a model.**

#### Packages used for this project include:
- Tensorflow
- Keras
- Numpy
- Scikit-learn
- Matplotlib
- Opendatasets
<br/>

#### Data
The [data](https://www.kaggle.com/datasets/gpiosenka/sports-classification) used for this project was gotten from Kaggle. Opendatasets was used to download the data to a local directory.<br/>
The Images in data were gathered from internet searches. All images were then resized to 224 X224 X 3 and converted to jpg format. 
<br/>

#### Model Architecture

![model](https://github.com/Jeremyugo/Sports-classifier-using-ResNet-50/assets/36512525/e37a4f8e-ea15-40e0-aaa0-23698bb1e968)

#### Model learning curves

![image](https://github.com/Jeremyugo/Sports-classifier-using-ResNet-50/assets/36512525/c8bb5732-d147-401d-83f6-aacde42b20e8)

#### Model confusion matrix
![image](https://github.com/Jeremyugo/Sports-classifier-using-ResNet-50/assets/36512525/52bd4151-458c-46a2-9bed-d1ca509447b6)


#### Model predictions

![image](https://github.com/Jeremyugo/Sports-classifier-using-ResNet-50/assets/36512525/55b518d6-c5ba-41e5-a55d-18b08d772a11)

#### Wrong model predictions
![image](https://github.com/Jeremyugo/Sports-classifier-using-ResNet-50/assets/36512525/57782680-1e15-454f-bd38-43c1e6d89f51)
