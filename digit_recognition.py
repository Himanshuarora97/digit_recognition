from __future__ import print_function
# get the labelled dataset
from keras.datasets import mnist
# models
from keras.models import Sequential
# layers 
# Dense -  fully connected layers
# Dropout - It is a regularization technique
# Flatten - To convert the output of CNN to 1D feature vector
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# keras backend
from keras import backend as K
# util to convert category feature to one-hot features
from keras import utils
# graph utility
import matplotlib.pyplot as plt


# Neural Network Hyperparameter
batch_size = 64
num_classes = 10 # 0 - 9
epochs = 1

# image dimentions 
rows, cols = 28, 28

# get the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)

# change data type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
'''RGB (Red, Green, Blue) are 8 bit each.
The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
The combination range is 256*256*256.
By dividing by 255, the 0-255 range can be described with a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF)
'''
X_train /= 255
X_test /= 255
print("X_train shape :",X_train.shape)
print("X_test shape :",X_test.shape)
print(input_shape)
# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# model
model = Sequential()
# Convolutional layer
model.add(Conv2D(32, 
				kernel_size= (3,3),
	 			activation = 'relu',
	 			input_shape = input_shape))
model.add(Conv2D(64,
				kernel_size = (3,3),
				activation = 'relu'))
# max pooling
model.add(MaxPooling2D(pool_size = (2,2)))
# regularization
model.add(Dropout(0.2))
# convert the high dimenstion to 1D feature using Flatten
model.add(Flatten())
# adding Dense layers (fully connected layers)
model.add(Dense(128, activation = 'relu'))
# regularization again
model.add(Dropout(0.35))
# output using softmax for output probabilities
model.add(Dense(num_classes, activation = 'softmax'))

# compile model
# categorical_crossentropy since this is a multiclassification problem 
# optimizer - Adam (can use AdaDelta or AdaGrad)
# since this is classification task accuracy is good enough
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

# print the model summary
print(model.summary())

# train the model
history = model.fit(X_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		verbose=True,
		validation_data=(X_test, y_test))

# graph
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('loss.png')


# get the evaluate score
test_score = model.evaluate(X_test, y_test, verbose=False)
print("Loss : ",test_score[0])
print("Accuracy : ",test_score[1])

# Now save the model
model_json = model.to_json()
with open('model.json','w') as f:
	f.write(model_json)

# save the weights to HDF5
model.save_weights('model.h5')
print("Model Saved Successfully!!!")



