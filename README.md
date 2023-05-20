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
