################################################################################
#
# LOGISTICS
#
#    Name: Raisaat Rashid
#    Net ID: rar150430
#
# FILE
#
#    nn.py
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my nn.py code:
#
#       The NN is made up of classes of layers - MatrixMulAndAddition, Relu and Softmax. 
#       Each layer class has its own forward and back prop functions. A final NN class 
#       puts all the layers together with the required dimensions. It passes the input 
#       through the forward pass of each layer sequentially as well as calls the back-prop
#       function of each layer during back propagation in the reverse direction.
#
#       Forward path code: A call to the forward function of the NN class first normalizes the 
#       input and vectorizes it. It then passes the input to a for loop that goes through each 
#       layer in the network in order and calls each individual layer's forward function, 
#       starting with the MatrixMulAndAdd layer. The MatrixMulAdd layer's forward function 
#       multiplies the input vector with the weights (which are initially assigned random values)
#       and adds biases to the result (intially assigned to 0). The Relu layer's forward function
#       takes an input vector and applies the Relu function to it pointwise. The softmax layer's 
#       forward function takes an input vector and  applies the softmax function to it pointwise. 
#       Each layer's forward function stores the input in a cache and returns the result of the 
#       forward pass. After forward pass through each layer, the nn calculates the loss between 
#       the actual output and the network's final output and returns that loss.
#       
#       Error code: Cross entropy was used to calculate the error between the true labels
#       and the predicted labels
#
#       Backward path code: A call to the backward function of the NN class first calculates the 
#       derivative of the cross entropy (the loss) with repect to the network's output and passes 
#       that to a for loop that goes through each layer in the network in reverse and calls each 
#       individual layer's backward function, starting with the softmax layer. Each layer's backward 
#       function takes the derivative of the loss with respect to the layer's ouput (dLoss/dOut) 
#       and the learning rate as parameters and calculates the derivative of the loss w.r.t. the 
#       previously stored input to the layer (dLoss/dIn) using dLoss/dOut and the derivative of 
#       the output w.r.t. the previously stored input to the layer (dOut/dIn); dLoss/dIn = dLoss/dOut * dOut/dIn.
#       The layer finally returns dLoss/dIn, which gets passed to the next layer in line as dLoss/dOut. 
#
#       Weight update code: If a layer has learnable parameters (the multiplication and bias layer), the 
#       layer's backward function also calculates the derivative of the loss w.r.t. each of these parameters 
#       (dLoss/dParam) using dLoss/dOut and the derivative of the output w.r.t. the parameters (dOut/dParam); 
#       dLoss/dParam = dLoss/dOut * dOut/dParam. It uses stochastic gradient descent to update the parameters
#       using dLoss/dParam and the learning rate that is passed to it: Param = Param - lr * dLoss/dParam.
#
#    2. Accuracy display:
#       
#       Epoch  0 lr = 0.000010 avg loss = 2.302501 accuracy = 11.35
#       Epoch  1 lr = 0.000340 avg loss = 2.301507 accuracy = 11.35
#       Epoch  2 lr = 0.000670 avg loss = 2.196799 accuracy = 32.89
#       Epoch  3 lr = 0.001000 avg loss = 0.698190 accuracy = 86.14
#       Epoch  4 lr = 0.000952 avg loss = 0.348741 accuracy = 91.13
#       Epoch  5 lr = 0.000811 avg loss = 0.233470 accuracy = 93.36
#       Epoch  6 lr = 0.000592 avg loss = 0.181167 accuracy = 94.56
#       Epoch  7 lr = 0.000316 avg loss = 0.155026 accuracy = 95.35
#       Epoch  8 lr = 0.000010 avg loss = 0.144924 accuracy = 95.49
#       
#       Final accuracy of test set = 95.49
#
#    3. Performance display:
#       
#       Total execution time: 66.81339806715647 minutes
#       
#       Per layer information
#       Division by 255.0: input size = 28 x 28 x 1, output size = 28 x 28 x 1
#       Vectorization: input size = 28 x 28 x 1, output size = 784 x 1 
#       Matrix Mulitplication: input size = 784 x 1, output size = 1000 x 1, parameter size = 784 x 1000, MACs = 784000
#       Addition: input size = 1000 x 1, output size = 1000 x 1, parameter size = 1000 x 1
#       ReLU: input size = 1000 x 1, output size = 1000 x 1
#       Matrix Multiplication: input size = 1000 x 1, output size = 100 x 1, parameter size = 1000 x 100, MACs = 100000
#       Addition: input size = 100 x 1, output size = 100 x 1, parameter size = 100 x 1
#       ReLU: input size = 100 x 1, output size = 100 x 1
#       Matrix Multiplication: input size = 100 x 1, output size = 10 x 1, parameter size = 100 x 10, MACs = 1000
#       Addition: input size = 10 x 1, output size = 10 x 1, parameter size = 10 x 1
#       Softmax: input size = 10 x 1, output size = 10 x 1
#
#    4. Extra features:
#       
#       One call to the NN's forward function (and back prop function) auto-generates the 
#       forward (and back prop) function of each layer. Each layer is flexible in the 
#       sense that it can be created with custom input and output dimensions.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

#
# you should not need any import beyond the below
# PyTorch, TensorFlow, ... is not allowed
#

import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

################################################################################
#
# YOUR CODE GOES HERE
#
################################################################################
# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 3
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 6
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

class Relu:
  # Relu layer to perform relu operation on inputs
  def forward(self, input):
    '''
    Executes a forward pass of the relu layer on the input.
    '''
    self.cache = input
    return np.maximum(0, input)

  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the relu layer.
    '''
    dOut_dIn = self.cache
    dOut_dIn[dOut_dIn <= 0] = 0
    dOut_dIn[dOut_dIn > 0] = 1

    return dLoss_dOut * dOut_dIn # return the dLoss/dIn where In is the input to the layer

class MatrixMulAndAdd:
  # Layer to perform matrix multiplication and addition on 2D inputs
  def __init__(self, input_dim, out_dim):
    self.weights = np.random.randn(input_dim, out_dim) / input_dim
    self.bias = np.zeros(out_dim)
  
  def forward(self, input):
    '''
    Executes a forward pass of the matrix multiplication and addition layer on the input.
    '''
    self.cache = input
    return np.dot(input, self.weights) + self.bias
  
  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the matrix multiplication and addition layer.
    '''
    dOut_dW = self.cache
    dOut_dIn = self.weights

    # Calculate the derivatives
    dLoss_dW = dOut_dW[np.newaxis].T @ dLoss_dOut[np.newaxis]
    dLoss_dIn = dOut_dIn @ dLoss_dOut

    # update weights and bias
    self.weights = self.weights - lr * dLoss_dW
    self.bias = self.bias - lr * dLoss_dOut

    return dLoss_dIn

class Softmax:
  # Softmax layer to perform softmax on the input  
  def forward(self, input):
    '''
    Executes a forward pass of the softmax layer on the input.
    '''
    self.cache = input
    exp = np.exp(input)
    return exp / np.sum(exp, axis=0)

  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the softmax layer.
    '''
    for i, derivative in enumerate(dLoss_dOut):
      if derivative == 0:
        continue

      t_exp = np.exp(self.cache)
      S = np.sum(t_exp)

      # Compute the derivatives
      dOut_dIn = -t_exp[i] * t_exp / (S ** 2)
      dOut_dIn[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
      return derivative * dOut_dIn

class NeuralNet:
  # A Neural Network
  def __init__(self):
    self.layers = [          
        MatrixMulAndAdd(784, 1000),
        Relu(),
        MatrixMulAndAdd(1000, 100),
        Relu(),
        MatrixMulAndAdd(100, 10),
        Softmax()
    ]
    self.cache = {}
  
  def forward(self, X, label):
    '''
    Executes a forward pass of all the layers on the input.
    '''
    X = np.divide(X, 255.0).flatten() # normalize and vectorize the input
    self.true_label = label

    for layer in self.layers: # do a forward pass
      X = layer.forward(X)
    
    self.Yh = X
    loss = -np.log(self.Yh[label]) # compute the loss

    return loss
  
  def backward(self, lr):
    '''
    Executes a backward pass of all the layers.
    '''

    # Calculate dLoss/dYh
    derivative = np.zeros(10)
    derivative[self.true_label] = -1 / self.Yh[self.true_label]

    for layer in reversed(self.layers): # do a backward pass
      derivative = layer.backward(derivative, lr)

  def predict(self, X, label):
    '''
    Makes a prediction of the output label given an input
    '''
    loss = self.forward(X, label)
    prediction = np.argmax(self.Yh)
    return prediction

# learning rate schedule
def lr_schedule(epoch):
  # linear warmup followed by cosine decay
  if epoch < TRAINING_LR_INIT_EPOCHS:
    lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
  else:
    lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL
  return lr

# start epoch
start_epoch = 0

cnn = NeuralNet() # Define an NN
epochs = []
accuracies = []
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):
  epochs.append(epoch)

  training_loss = 0.0
  accuracy_sum = 0.0
  lr = lr_schedule(epoch)

  # train
  for i in range(DATA_NUM_TRAIN):
    loss = cnn.forward(train_data[i].reshape(DATA_ROWS, DATA_COLS, 1), train_labels[i])
    cnn.backward(lr)
    training_loss = training_loss + loss
  
  # test
  test_correct = 0.0
  for j in range(DATA_NUM_TEST):
     test_correct = test_correct + (cnn.predict(test_data[j].reshape(DATA_ROWS, DATA_COLS, 1), test_labels[j]) == test_labels[j])
  
  accuracy = 100.0*(test_correct/DATA_NUM_TEST)
  accuracies.append(accuracy)
  # epoch statistics
  print('Epoch {0:2d} lr = {1:8.6f} avg loss = {2:8.6f} accuracy = {3:5.2f}'.format(epoch, lr, training_loss/DATA_NUM_TRAIN, accuracy))

################################################################################
#
# DISPLAY
#
################################################################################

# test with the final model
test_correct = 0.0
predictions = []
for j in range(DATA_NUM_TEST):
  prediction = cnn.predict(test_data[j].reshape(DATA_ROWS, DATA_COLS, 1), test_labels[j])
  test_correct = test_correct + (prediction == test_labels[j])
  predictions.append(prediction)

# test set statistics
print('Final accuracy of test set = {0:5.2f}'.format((100.0*test_correct/DATA_NUM_TEST)))
plt.plot(epochs, accuracies)
plt.xlabel("epoch")
plt.ylabel("Accuracy (%)")

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(predictions[i]))
    plt.imshow(img, cmap='Greys')
plt.show()

print("\nTotal execution time: %s minutes" % ((time.time() - start_time)/60))