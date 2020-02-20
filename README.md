# E0 250 - Deep Learning
## FNN and CNN for Fashion-MNIST

### Fashion-MNIST dataset
The **Fashion-MNIST** is a dataset of Zalando’s article images, consisting of a training set of **60,000** examples and a test set of **10,000** examples. Each example is a **28x28** grayscale image, associated with a label from **10** classes. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have **785** columns. The ﬁrst column consists of the *class labels **(0-9)***, and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

### Loss functions used
- FNN: **Mean Squared Error.**
- CNN: **Cross Entropy.**

### Results
Implemented a fully connected **feedforward neural network** (FNN) and a **convolutional neural network** (CNN) for the classiﬁcation task of recognizing an image and identifying it as one of the ten possible classes in Fashion-MNIST clothing images dataset using the PyTorch library. Both the networks were trained for exactly 90 epochs.  

To view the network architecture, check the **classes** folder. The trained models have been saved in the **models** folder, where the code for training the models can also be found.  

The files **plot_cnn** and **plot_fnn** show the plots of *loss* vs the *number of epochs* for the respective networks.  

FNN: Accuracy of **90.33 %** on the test set.  
CNN: Accuracy of **91.07 %** on the test set.
