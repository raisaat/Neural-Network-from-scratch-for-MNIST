# Neural-Network-from-scratch-for-MNIST
This project creates a neural network from scratch with no external ML-specific libraries like PyTorch or Tensorflow to provide an insight into how traditional neural networks operate on a deep level. The neural network is made up of multiple _layers_ which, in the _forward pass_, map input data to intermediate feature maps to output class probability masses. An error is calculated by comparing the output class probability with the true class probability (which is typically 1). The sensitivity of this error with respect feature maps and with respect to the network parameters is calculated and _backpropagated_ to the network in the _backword pass_ to update the network parameters. This is how the network is trained. After training, an input only goes through the forward pass. 

## Summary of the code:
The NN is made up of classes of layers - MatrixMulAndAddition, Relu and Softmax. Each layer class has its own forward and back prop functions. A final NN class puts all the layers together with the required dimensions. It passes the input through the forward pass of each layer sequentially as well as calls the back-prop function of each layer during back propagation in the reverse direction.

**Forward path code:** A call to the forward function of the NN class first normalizes the input and vectorizes it. It then passes the input to a for loop that goes through each layer in the network in order and calls each individual layer's forward function, starting with the MatrixMulAndAdd layer. The MatrixMulAdd layer's forward function multiplies the input vector with the weights (which are initially assigned random values) and adds biases to the result (intially assigned to 0). The Relu layer's forward function takes an input vector and applies the Relu function to it pointwise. The softmax layer's forward function takes an input vector and  applies the softmax function to it pointwise. Each layer's forward function stores the input in a cache and returns the result of the forward pass. After forward pass through each layer, the nn calculates the loss between the actual output and the network's final output and returns that loss.

**Error code:** Cross entropy was used to calculate the error between the true labels and the predicted labels

**Backward path code:** A call to the backward function of the NN class first calculates the derivative of the cross entropy (the loss) with repect to the network's output and passes that to a for loop that goes through each layer in the network in reverse and calls each individual layer's backward function, starting with the softmax layer. Each layer's backward function takes the derivative of the loss with respect to the layer's ouput (dLoss/dOut) and the learning rate as parameters and calculates the derivative of the loss w.r.t. the previously stored input to the layer (dLoss/dIn) using dLoss/dOut and the derivative of the output w.r.t. the previously stored input to the layer (dOut/dIn); dLoss/dIn = dLoss/dOut * dOut/dIn. The layer finally returns dLoss/dIn, which gets passed to the next layer in line as dLoss/dOut. 

**Weight update code:** If a layer has learnable parameters (the multiplication and bias layer), the layer's backward function also calculates the derivative of the loss w.r.t. each of these parameters (dLoss/dParam) using dLoss/dOut and the derivative of the output w.r.t. the parameters (dOut/dParam); dLoss/dParam = dLoss/dOut * dOut/dParam. It uses stochastic gradient descent to update the parameters using dLoss/dParam and the learning rate that is passed to it: Param = Param - lr * dLoss/dParam.

## Architecture:

**Division by 255.0:** input size = 28 x 28 x 1, output size = 28 x 28 x 1  
**Vectorization:** input size = 28 x 28 x 1, output size = 784 x 1   
**Matrix Mulitplication:** input size = 784 x 1, output size = 1000 x 1, parameter size = 784 x 1000, MACs = 784000  
**Addition:** input size = 1000 x 1, output size = 1000 x 1, parameter size = 1000 x 1  
**ReLU:** input size = 1000 x 1, output size = 1000 x 1  
**Matrix Multiplication:** input size = 1000 x 1, output size = 100 x 1, parameter size = 1000 x 100, MACs = 100000  
**Addition:** input size = 100 x 1, output size = 100 x 1, parameter size = 100 x 1  
**ReLU:** input size = 100 x 1, output size = 100 x 1  
**Matrix Multiplication:** input size = 100 x 1, output size = 10 x 1, parameter size = 100 x 10, MACs = 1000  
**Addition:** input size = 10 x 1, output size = 10 x 1, parameter size = 10 x 1  
**Softmax:** input size = 10 x 1, output size = 10 x 1  

## Accuracy Display:

```
Epoch  0 lr = 0.000010 avg loss = 2.302501 accuracy = 11.35
Epoch  1 lr = 0.000340 avg loss = 2.301507 accuracy = 11.35
Epoch  2 lr = 0.000670 avg loss = 2.196799 accuracy = 32.89
Epoch  3 lr = 0.001000 avg loss = 0.698190 accuracy = 86.14
Epoch  4 lr = 0.000952 avg loss = 0.348741 accuracy = 91.13
Epoch  5 lr = 0.000811 avg loss = 0.233470 accuracy = 93.36
Epoch  6 lr = 0.000592 avg loss = 0.181167 accuracy = 94.56
Epoch  7 lr = 0.000316 avg loss = 0.155026 accuracy = 95.35
Epoch  8 lr = 0.000010 avg loss = 0.144924 accuracy = 95.49
      
Final accuracy of test set = 95.49
```
## Performance Display:

`Total execution time: 66.81339806715647 minutes`
