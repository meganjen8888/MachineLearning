Welcome to your third programming assignment! We have so far learned how to implement a linear regression model for regression task and implement a logistic regression model for classification task. The essential steps in the implementations are:

1. Define your hypothesis, for example $h_w(x) = w^Tx$ for linear regression model where both $w$ and $x$ are vectors;

2. Define your cost or total loss function, and given the training data, calculate the cost or total loss comparing with the true labels or values of the training examples;

3. Use gradient descent to optimize the cost function, and find the parameters in your hypothesis.

4. Use the hypothesis with the found parameters to predict on unseen data.

And we have introduced machine learning library scikit-learn,which has many models already built for you, we can just import linear regression or logistic regression model and fit our training data to learn a model to predict on unseen data. But for deep learning, scikit-learn is not suitable anymore (non GPU optimizations), so we introduced you popular deep learning frameworks Pytorch and TensorFlow.

In this assignment, you will first see how to use scikit-learn to train logistic regression model for assignment 2 data (student admission), and then practice with Pytorch basics and then use Pytorch and TensorFlow to build a neural network model for a house price prediction (regression task) and a neural network model for predicting breast cancer (classification task).

<span style="color:red">To submit</span>: You will need to submit this notebook with proper naming <span style="color:blue">Assignment3_FirstName_LastName.ipynb</span> to blackboard.

<span style="color:red">Important</span>: <span style="color:blue">Remember to restart the kernel everytime you modify your model or training steps</span> Otherwise you may see some strange behavior which may be caused by "model" training being accumulated.

Megan Jen -- March 3, 2022


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

We will again first load the data from assignment 2.


```python
# loading file to array data
data = np.loadtxt('a2data1.txt', delimiter=",")
print(data.shape)
print(data[:5, :])

# get training examples features into X and label to y
X = data[:, :2]
y = data[:, -1]
print(X.shape)
print(y.shape)
```

    (100, 3)
    [[34.62365962 78.02469282  0.        ]
     [30.28671077 43.89499752  0.        ]
     [35.84740877 72.90219803  0.        ]
     [60.18259939 86.3085521   1.        ]
     [79.03273605 75.34437644  1.        ]]
    (100, 2)
    (100,)


## 1. Logistic regression with Scikit-learn
We will first see how to use logistic regression model from scikit-learn library.


```python
from sklearn.linear_model import LogisticRegression

# create model and fit in the data
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# calcute the training accuracy, 
# you should see 0.89 which is same as your model in assignment 2
train_accuracy = model.score(X, y)
print("Train accuracy is: {}".format(train_accuracy))
```

    Train accuracy is: 0.89


## 2. Three-layer neural network model with Pytorch

Now we will create a three-layer (two hidden layers) neural network model with this dataset again, and compare the training accuracy with Logistic regression model. Before we begin, let's practice with basics of Pytorch: torch.tensor, torch. autograd, torch.nn.Module. The official documentation is <a href="https://pytorch.org/tutorials/beginner/basics/intro.html">here</a>.


```python
# import torch first and print its version
import torch
print("torch version:", torch.__version__) 

# add a random seed so that everytime you run the model it will give similar results
torch.manual_seed(10)
```

    torch version: 1.10.2





    <torch._C.Generator at 0x7fa2a2a93330>



### 2.1 Pytorch Tensor
We have discussed that pytorch tensor is just like numpy array but can run on GPUs. So many operations of tensors are similar to numpy arrays, and we can create tensors from list, numpy arrays and convert tensors to numpy arrays.

#### 2.1.1 Initialize a Tensor
Now let's see ways to initialize a Tensor.

##### 2.1.1.1 Directly from data


```python
# 1. Tensors can be created directly from data. 
# The data type is automatically inferred.

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)
```

    tensor([[1, 2],
            [3, 4]])


##### 2.1.1.2 From Numpy arrays


```python
# 2. Tensors can be created from NumPy arrays.
np_array = np.array([[1, 2],[3, 4]])
x_np = torch.from_numpy(np_array)
print(x_np)
```

    tensor([[1, 2],
            [3, 4]])



```python
# tensor to numpy
x_array = x_np.numpy()
print(x_array)
```

    [[1 2]
     [3 4]]


##### 2.1.1.3 From another tensor


```python
# 3. From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor: \n {x_rand} \n")
```

    Ones Tensor: 
     tensor([[1, 1],
            [1, 1]]) 
    
    Random Tensor: 
     tensor([[0.4581, 0.4829],
            [0.3125, 0.6150]]) 
    


#### 2.1.2 Operations on Tensors 

Tensor operations are very similiar to numpy arrays. There built-in functions for tensor multiplication, element-wise mulitplication and so on. You can move a tensor across devices using .to():



```python
# We move our tensor to the GPU if available
t = torch.randn(2)

device = "cuda" if torch.cuda.is_available() else "cpu"
t = t.to(device)

print(" device:\t'{}'".format(t.device))
print(t)
```

     device:	'cpu'
    tensor([-0.3023, -1.2277])


#### 2.1.2.1 Exercise

And some common operations on tensors: 
<tt><p>
torch.matmul(a, b) # multiples torch tensors a and b

a * b      # element-wise multiplication between two torch tensors

torch.eye(n) # creates an identity torch tensor with shape (n, n)

torch.zeros(n, m) # creates a torch tensor of zeros with shape (n, m)

torch.ones(n, m) # creates a torch tensor of ones with shape (n, m)

torch.rand(n, m) # creates a random torch tensor with shape (n, m)
<p></tt>

Now practice with operations with defining two tensors first.


```python
# create tensor of ones with shape (2, 2), assign to tensor_a
tensor_a = torch.ones(2, 2)
# create tensor_b 
tensor_b = tensor_a * 2
print(tensor_a, tensor_a.shape)
print(tensor_b)

```

    tensor([[1., 1.],
            [1., 1.]]) torch.Size([2, 2])
    tensor([[2., 2.],
            [2., 2.]])



```python
# Exercise create a random tensor with shape (2, 4)
# START CODE HERE
tensor_c = torch.rand(2, 4)
# END CODE HERE
```


```python
# Exercise torch.matmul(..) function with tensor_a, and tensor_b
# START CODE HERE
torch.matmul(tensor_a, tensor_b)
# END CODE HERE
```




    tensor([[4., 4.],
            [4., 4.]])



#### 2.1.3 Tensor.autograd

We have given the examples for autograd package in class, the following code part is from the slide:


```python
# manually calculate gradients

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

```


```python
# With torch.autograd to automatical gradients

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    loss.backward()
    
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
```

#### 2.1.4 Torch.nn

We also talked about high level wrapper torch.nn.Module, and how to define a customized Module to define your neural network model, and how to use torch.optim package to optimize the model. The following code is from the slides in Lecture 7.



```python

class TwoLayerNet(torch.nn.Module): 
    def __init__(self, D_in, H, D_out): 
        super(TwoLayerNet, self).__init__() 
        self.linearl = torch.nn.Linear(D_in, H) 
        self.linear2 = torch.nn.Linear(H, D_out) 
    
    def forward(self, x): 
        h_relu = self.linearl(x).clamp(min=0) 
        ypred = self.linear2(h_relu) 
        return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 
x = torch.randn(N, D_in) 
y = torch.randn(N, D_out) 

model = TwoLayerNet(D_in, H, D_out) 

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) 

for t in range(500): 
    ypred = model(x) 
    loss = torch.nn.functional.mse_loss(ypred, y) 
    loss.backward() 
    optimizer.step() 
    optimizer.zero_grad() 

```

### 2.2 Three-layer neural network for assignment2 data

Now you can follow the example in the above cell to define a neural network model for assignment 2 data. We can build a three-layer neural network model with 16 hidden units for 1st hidden layer, and  8 hidden units on the 2nd hidden layer, and we want the output layer to have sigmoid() function as activation function. First, you will need to figure what's input dimension is, that is how many features each training example has. Recall that we are given two exam scores of one student, so there two features, you don't need to worry about adding the bias or interception anymore, pytorch will do that for you. 


```python
# loading file to array data
data = np.loadtxt('a2data1.txt', delimiter=",")

# get training examples features into X and label to y
X = data[:, :2]
y = data[:, -1]
print(X.shape)
print(y.shape)
```

    (100, 2)
    (100,)


First, we create a torch.nn.Module that define this three-layer neural network. We want this three-layer neural network model to have, 16 hidden units for 1st hidden layer, and 8 hidden units on the 2nd hidden layer, and we want the output layer to have sigmoid() function as activation function. We can use torch.nn.Sigmoid() for this.

You can refer to the two-layer model above to define this class.


```python
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_1, H_2, D_out): 
        super(ThreeLayerNet, self).__init__() 
        self.linear1 = torch.nn.Linear(D_in, H_1) 
        self.linear2 = torch.nn.Linear(H_1, H_2)
        
        # START CODE HERE (2 line)
        self.linear3 = torch.nn.Linear(H_2, D_out) 
        self.sigmoid = torch.nn.Sigmoid()
        # END CODE HERE
    
    def forward(self, x): 
        h1_relu = self.linear1(x).clamp(min=0) 
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        
        # STAET CODE HERE (1 ~ 2 line)
        ypred = self.sigmoid(self.linear3(h2_relu))
        # END CODE HERE
        
        return ypred 
```

Next, we will define input , 1st hidden, 2nd hidden, output dimensions of the model D_in (should be 2), H_1 (16 hidden units), H_2 (8 hidden_units), D_out (should be 1).


```python
# Define input, 1st hidden, 2nd hidden, output dimensions of the model D_in, H_1, H_2, D_out

# STAET CODE HERE (1 line)
D_in, H_1, H_2, D_out = 2, 16, 8, 1
# END CODE HERE
```

Next, we will initialize a model from the class ThreeLayerNet with dimensions defined above. 



```python
# create a model from class ThreeLayerNet(..) and name it model_3layer

# START CODE HERE (1 line) 
model_3layer = ThreeLayerNet(D_in, H_1, H_2, D_out)
# END CODE HERE
```

Next, we can use torch.optim to optimize the model. One thing different from our example above is the loss function. Since we are doing classification, we will use cross-entropy loss, more specifically the binary cross-entropy loss that is torch.nn.BCELoss(),  which is essentially the loss function  we used for logistic regression. For multi-class, you will use torch.nn.CrossEntropyLoss().


```python
# converting input to tensors, and converting to float type because of tensor weights type is float 
tensor_X = torch.from_numpy(X).float()

# reshape true labels in y to be same shape as pytorch model's output
tensor_y = torch.from_numpy(y.reshape(100, 1)).float()
tensor_y.shape
```




    torch.Size([100, 1])




```python
optimizer = torch.optim.Adam(model_3layer.parameters(), lr=1e-2) 

for t in range(200):
    # START CODE HERE (1 lines)
    ypred = model_3layer(tensor_X)
    # END CODE HERE
     
    loss_fn = torch.nn.BCELoss() #loss function
    loss = loss_fn(ypred, tensor_y) #calculating loss
    
    # START CODE HERE (2 lines)
    #back propogation
    loss.backward() #back propogation
    optimizer.step() #how to do back propogation in pytorch. this is updating parameters
    # END CODE HERE
     
    optimizer.zero_grad() 


```


```python
from sklearn.metrics import accuracy_score

# get predictions of the training examples
# and calculate training accuracy
y_hat = ypred.detach().numpy()
train_pred_labels = y_hat >= 0.5
train_accuracy = accuracy_score(train_pred_labels, y)
print("Training accuracy is: {} %".format(train_accuracy * 100))
```

    Training accuracy is: 92.0 %


|Expected|
|:-|
|Training accuracy is: 91.0 % (or somewhere over 90%)|

## 3. TensorFlow exercise

In this section 3, you will practice with TensorFlow. Taken the slides code as examples, and also refer to the official documentation <a href="https://pytorch.org/tutorials/beginner/basics/intro.html">here</a> to build a same three-layer neural network. Train it with the assignment 2 data again, and see whether you will get the same result as the pytorch one or not. 


```python
import tensorflow as tf
print(tf.__version__)
```

    2.8.0


### 3.1 TensorFlow basics
Below are some of basic operations on tensors in tensorflow.


```python
# Create a constant tensor A
A = tf.constant([[1,2],
                  [3,4]])
# Create a Variable tensor B
B = tf.Variable([[5,6],
                 [7,8]])
print(A)
print(B)
```

    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32)
    <tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
    array([[5, 6],
           [7, 8]], dtype=int32)>



```python
# Concatenate Two Tensor along the sencond dimension
C = tf.concat([A,B], axis=1)
C
```




    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[1, 2, 5, 6],
           [3, 4, 7, 8]], dtype=int32)>




```python
# Concatenate Two Tensor along the first dimension
D = tf.concat([A,B], axis=0)
D
```




    <tf.Tensor: shape=(4, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8]], dtype=int32)>




```python
# Creates a tensor with all elements set to zero.
E = tf.zeros(shape=[4,3],dtype=tf.float16)
E
```




    <tf.Tensor: shape=(4, 3), dtype=float16, numpy=
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]], dtype=float16)>




```python
# Creates a tensor with all elements set to one.
F = tf.ones(shape=[4,3],dtype=tf.int32)
F
```




    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]], dtype=int32)>




```python
# Creates a tensor with elements sampled from a uniform distribution.
G = tf.random.uniform(shape=[2,6])
G
```




    <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0.09022141, 0.24294996, 0.7336577 , 0.81594574, 0.21499443,
            0.6360214 ],
           [0.2569946 , 0.07807779, 0.8118094 , 0.16622162, 0.89384353,
            0.04577541]], dtype=float32)>




```python
# Typecast a tensor
I = H * 100.0
print(I)
J = tf.cast(I, dtype=tf.int32)
print(J)
```

    10000.0
    tf.Tensor(10000, shape=(), dtype=int32)



```python
# Matrix multiply
A = tf.constant([[1,2],
                  [3,4]])
x = tf.constant([[2],
                 [3]])
M = tf.matmul(A, x)
print(M)
```

    tf.Tensor(
    [[ 8]
     [18]], shape=(2, 1), dtype=int32)



```python
# Element-wise math operation
B = tf.constant([[-1,-2],
                 [-3,-4]])
N = tf.multiply(A,B)
print(N)
```

    tf.Tensor(
    [[ -1  -4]
     [ -9 -16]], shape=(2, 2), dtype=int32)


### 3.2 Load the data again and Build the model

See the slides code and also tutorial here for a text classification <a href="https://www.tensorflow.org/tutorials/keras/text_classification">here</a>.


```python
# loading file to array data
data = np.loadtxt('a2data1.txt', delimiter=",")

# get training examples features into X and label to y
X = data[:, :2]
y = data[:, -1]
print(X.shape)
print(y.shape)
```

    (100, 2)
    (100,)


Build the model.


```python
model_tf = tf.keras.Sequential()

model_tf.add(tf.keras.layers.Dense(16, input_shape=(2,), activation="relu"))
model_tf.add(tf.keras.layers.Dense(8, input_shape=(16,), activation="relu"))

# Fill the blank
model_tf.add(tf.keras.layers.Dense(1, input_shape=(8,), activation="sigmoid"))

optimizer = tf.optimizers.Adam(1e-2)

model_tf.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),optimizer=optimizer, metrics=tf.metrics.BinaryAccuracy())

history = model_tf.fit(X, y, epochs=400, batch_size=100)
```

    Epoch 1/400
    1/1 [==============================] - 1s 1s/step - loss: 2.5740 - binary_accuracy: 0.6000
    Epoch 2/400
    1/1 [==============================] - 0s 11ms/step - loss: 1.7594 - binary_accuracy: 0.6000
    Epoch 3/400
    1/1 [==============================] - 0s 84ms/step - loss: 1.1981 - binary_accuracy: 0.6000
    Epoch 4/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.8940 - binary_accuracy: 0.6000
    Epoch 5/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.7450 - binary_accuracy: 0.6000
    Epoch 6/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.6467 - binary_accuracy: 0.6200
    Epoch 7/400
    1/1 [==============================] - 0s 22ms/step - loss: 0.6698 - binary_accuracy: 0.5800
    Epoch 8/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.7386 - binary_accuracy: 0.4600
    Epoch 9/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.7555 - binary_accuracy: 0.4500
    Epoch 10/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.7389 - binary_accuracy: 0.4500
    Epoch 11/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.7137 - binary_accuracy: 0.4400
    Epoch 12/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.6851 - binary_accuracy: 0.4700
    Epoch 13/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.6732 - binary_accuracy: 0.7100
    Epoch 14/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.6728 - binary_accuracy: 0.8000
    Epoch 15/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.6707 - binary_accuracy: 0.6700
    Epoch 16/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.6673 - binary_accuracy: 0.6200
    Epoch 17/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6635 - binary_accuracy: 0.6100
    Epoch 18/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6594 - binary_accuracy: 0.6100
    Epoch 19/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6549 - binary_accuracy: 0.6100
    Epoch 20/400
    1/1 [==============================] - 0s 76ms/step - loss: 0.6501 - binary_accuracy: 0.6100
    Epoch 21/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.6448 - binary_accuracy: 0.6300
    Epoch 22/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.6394 - binary_accuracy: 0.6300
    Epoch 23/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.6331 - binary_accuracy: 0.6300
    Epoch 24/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.6250 - binary_accuracy: 0.6600
    Epoch 25/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6211 - binary_accuracy: 0.6700
    Epoch 26/400
    1/1 [==============================] - 0s 48ms/step - loss: 0.6197 - binary_accuracy: 0.6500
    Epoch 27/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.6163 - binary_accuracy: 0.6500
    Epoch 28/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.6106 - binary_accuracy: 0.6700
    Epoch 29/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.6057 - binary_accuracy: 0.6600
    Epoch 30/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.6045 - binary_accuracy: 0.6400
    Epoch 31/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.6048 - binary_accuracy: 0.6300
    Epoch 32/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.6047 - binary_accuracy: 0.6300
    Epoch 33/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6031 - binary_accuracy: 0.6400
    Epoch 34/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6006 - binary_accuracy: 0.6400
    Epoch 35/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.5993 - binary_accuracy: 0.6600
    Epoch 36/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.6000 - binary_accuracy: 0.6700
    Epoch 37/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.6001 - binary_accuracy: 0.6700
    Epoch 38/400
    1/1 [==============================] - 0s 21ms/step - loss: 0.5972 - binary_accuracy: 0.6700
    Epoch 39/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.5946 - binary_accuracy: 0.6700
    Epoch 40/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.5932 - binary_accuracy: 0.6600
    Epoch 41/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.5924 - binary_accuracy: 0.6400
    Epoch 42/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.5905 - binary_accuracy: 0.6400
    Epoch 43/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5878 - binary_accuracy: 0.6600
    Epoch 44/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5860 - binary_accuracy: 0.6700
    Epoch 45/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.5851 - binary_accuracy: 0.6800
    Epoch 46/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5842 - binary_accuracy: 0.6800
    Epoch 47/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5831 - binary_accuracy: 0.6800
    Epoch 48/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5819 - binary_accuracy: 0.6800
    Epoch 49/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.5803 - binary_accuracy: 0.6700
    Epoch 50/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.5799 - binary_accuracy: 0.6600
    Epoch 51/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.5796 - binary_accuracy: 0.6500
    Epoch 52/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.5787 - binary_accuracy: 0.6500
    Epoch 53/400
    1/1 [==============================] - 0s 6ms/step - loss: 0.5775 - binary_accuracy: 0.6700
    Epoch 54/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5765 - binary_accuracy: 0.6800
    Epoch 55/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.5760 - binary_accuracy: 0.6800
    Epoch 56/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.5752 - binary_accuracy: 0.6800
    Epoch 57/400
    1/1 [==============================] - 0s 50ms/step - loss: 0.5742 - binary_accuracy: 0.6800
    Epoch 58/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.5731 - binary_accuracy: 0.6800
    Epoch 59/400
    1/1 [==============================] - 0s 6ms/step - loss: 0.5720 - binary_accuracy: 0.6800
    Epoch 60/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.5710 - binary_accuracy: 0.6800
    Epoch 61/400
    1/1 [==============================] - 0s 54ms/step - loss: 0.5700 - binary_accuracy: 0.6800
    Epoch 62/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.5690 - binary_accuracy: 0.6800
    Epoch 63/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5681 - binary_accuracy: 0.6800
    Epoch 64/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5669 - binary_accuracy: 0.6800
    Epoch 65/400
    1/1 [==============================] - 0s 21ms/step - loss: 0.5661 - binary_accuracy: 0.6800
    Epoch 66/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5652 - binary_accuracy: 0.6800
    Epoch 67/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5644 - binary_accuracy: 0.6800
    Epoch 68/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5634 - binary_accuracy: 0.6800
    Epoch 69/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.5627 - binary_accuracy: 0.6800
    Epoch 70/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5618 - binary_accuracy: 0.6800
    Epoch 71/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5608 - binary_accuracy: 0.6800
    Epoch 72/400
    1/1 [==============================] - 0s 36ms/step - loss: 0.5598 - binary_accuracy: 0.6800
    Epoch 73/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.5590 - binary_accuracy: 0.6900
    Epoch 74/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5581 - binary_accuracy: 0.6900
    Epoch 75/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.5569 - binary_accuracy: 0.6900
    Epoch 76/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.5561 - binary_accuracy: 0.6800
    Epoch 77/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.5550 - binary_accuracy: 0.6800
    Epoch 78/400


    1/1 [==============================] - 0s 14ms/step - loss: 0.5540 - binary_accuracy: 0.6900
    Epoch 79/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5517 - binary_accuracy: 0.6900
    Epoch 80/400
    1/1 [==============================] - 0s 5ms/step - loss: 0.5510 - binary_accuracy: 0.6800
    Epoch 81/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.5453 - binary_accuracy: 0.6900
    Epoch 82/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5489 - binary_accuracy: 0.6700
    Epoch 83/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.5422 - binary_accuracy: 0.6800
    Epoch 84/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5445 - binary_accuracy: 0.6900
    Epoch 85/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.5432 - binary_accuracy: 0.6900
    Epoch 86/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.5388 - binary_accuracy: 0.6700
    Epoch 87/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.5429 - binary_accuracy: 0.6600
    Epoch 88/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.5349 - binary_accuracy: 0.6900
    Epoch 89/400
    1/1 [==============================] - 0s 6ms/step - loss: 0.5362 - binary_accuracy: 0.6900
    Epoch 90/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.5326 - binary_accuracy: 0.6900
    Epoch 91/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.5291 - binary_accuracy: 0.6900
    Epoch 92/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.5283 - binary_accuracy: 0.6900
    Epoch 93/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5246 - binary_accuracy: 0.7000
    Epoch 94/400
    1/1 [==============================] - 0s 68ms/step - loss: 0.5244 - binary_accuracy: 0.7100
    Epoch 95/400
    1/1 [==============================] - 0s 34ms/step - loss: 0.5163 - binary_accuracy: 0.6900
    Epoch 96/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.5211 - binary_accuracy: 0.6700
    Epoch 97/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.5114 - binary_accuracy: 0.6900
    Epoch 98/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.5112 - binary_accuracy: 0.7100
    Epoch 99/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.5029 - binary_accuracy: 0.6900
    Epoch 100/400
    1/1 [==============================] - 0s 34ms/step - loss: 0.4992 - binary_accuracy: 0.6800
    Epoch 101/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.4931 - binary_accuracy: 0.7100
    Epoch 102/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.4910 - binary_accuracy: 0.7700
    Epoch 103/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.4869 - binary_accuracy: 0.7700
    Epoch 104/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.4857 - binary_accuracy: 0.7300
    Epoch 105/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.4809 - binary_accuracy: 0.7500
    Epoch 106/400
    1/1 [==============================] - 0s 41ms/step - loss: 0.4814 - binary_accuracy: 0.7500
    Epoch 107/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.4755 - binary_accuracy: 0.7500
    Epoch 108/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.4756 - binary_accuracy: 0.7200
    Epoch 109/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.4702 - binary_accuracy: 0.7500
    Epoch 110/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.4700 - binary_accuracy: 0.7600
    Epoch 111/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.4647 - binary_accuracy: 0.7500
    Epoch 112/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.4626 - binary_accuracy: 0.7500
    Epoch 113/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.4594 - binary_accuracy: 0.7600
    Epoch 114/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.4555 - binary_accuracy: 0.7500
    Epoch 115/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.4537 - binary_accuracy: 0.7600
    Epoch 116/400
    1/1 [==============================] - 0s 45ms/step - loss: 0.4490 - binary_accuracy: 0.7500
    Epoch 117/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.4472 - binary_accuracy: 0.7600
    Epoch 118/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.4430 - binary_accuracy: 0.7500
    Epoch 119/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.4409 - binary_accuracy: 0.7800
    Epoch 120/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.4372 - binary_accuracy: 0.7700
    Epoch 121/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.4342 - binary_accuracy: 0.7700
    Epoch 122/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.4316 - binary_accuracy: 0.7800
    Epoch 123/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.4276 - binary_accuracy: 0.7900
    Epoch 124/400
    1/1 [==============================] - 0s 22ms/step - loss: 0.4250 - binary_accuracy: 0.7800
    Epoch 125/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.4222 - binary_accuracy: 0.8000
    Epoch 126/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.4183 - binary_accuracy: 0.8000
    Epoch 127/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.4159 - binary_accuracy: 0.8000
    Epoch 128/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.4131 - binary_accuracy: 0.7900
    Epoch 129/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.4091 - binary_accuracy: 0.8200
    Epoch 130/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.4060 - binary_accuracy: 0.8200
    Epoch 131/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.4034 - binary_accuracy: 0.8000
    Epoch 132/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.4000 - binary_accuracy: 0.8000
    Epoch 133/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.3961 - binary_accuracy: 0.8200
    Epoch 134/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.3926 - binary_accuracy: 0.8200
    Epoch 135/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.3896 - binary_accuracy: 0.8200
    Epoch 136/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.3872 - binary_accuracy: 0.8100
    Epoch 137/400
    1/1 [==============================] - 0s 29ms/step - loss: 0.3838 - binary_accuracy: 0.8300
    Epoch 138/400
    1/1 [==============================] - 0s 6ms/step - loss: 0.3808 - binary_accuracy: 0.8100
    Epoch 139/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.3777 - binary_accuracy: 0.8400
    Epoch 140/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.3748 - binary_accuracy: 0.8300
    Epoch 141/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.3725 - binary_accuracy: 0.8400
    Epoch 142/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.3715 - binary_accuracy: 0.8100
    Epoch 143/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.3729 - binary_accuracy: 0.8400
    Epoch 144/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.3752 - binary_accuracy: 0.7800
    Epoch 145/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.3653 - binary_accuracy: 0.8500
    Epoch 146/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.3547 - binary_accuracy: 0.8600
    Epoch 147/400
    1/1 [==============================] - 0s 21ms/step - loss: 0.3517 - binary_accuracy: 0.8700
    Epoch 148/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.3548 - binary_accuracy: 0.8600
    Epoch 149/400
    1/1 [==============================] - 0s 38ms/step - loss: 0.3549 - binary_accuracy: 0.8000
    Epoch 150/400
    1/1 [==============================] - 0s 38ms/step - loss: 0.3443 - binary_accuracy: 0.8700
    Epoch 151/400
    1/1 [==============================] - 0s 22ms/step - loss: 0.3398 - binary_accuracy: 0.9000
    Epoch 152/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.3416 - binary_accuracy: 0.8300
    Epoch 153/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.3407 - binary_accuracy: 0.8900
    Epoch 154/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.3339 - binary_accuracy: 0.8600


    Epoch 155/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.3274 - binary_accuracy: 0.8900
    Epoch 156/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.3277 - binary_accuracy: 0.9000
    Epoch 157/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.3296 - binary_accuracy: 0.8600
    Epoch 158/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.3239 - binary_accuracy: 0.8900
    Epoch 159/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.3170 - binary_accuracy: 0.8800
    Epoch 160/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.3143 - binary_accuracy: 0.8900
    Epoch 161/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.3144 - binary_accuracy: 0.9100
    Epoch 162/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.3132 - binary_accuracy: 0.8900
    Epoch 163/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.3075 - binary_accuracy: 0.9300
    Epoch 164/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.3026 - binary_accuracy: 0.9000
    Epoch 165/400
    1/1 [==============================] - 0s 99ms/step - loss: 0.3009 - binary_accuracy: 0.9000
    Epoch 166/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.3001 - binary_accuracy: 0.9400
    Epoch 167/400
    1/1 [==============================] - 0s 21ms/step - loss: 0.3000 - binary_accuracy: 0.9000
    Epoch 168/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2946 - binary_accuracy: 0.9300
    Epoch 169/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.2906 - binary_accuracy: 0.9000
    Epoch 170/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.2878 - binary_accuracy: 0.9100
    Epoch 171/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.2867 - binary_accuracy: 0.9400
    Epoch 172/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.2857 - binary_accuracy: 0.9100
    Epoch 173/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2842 - binary_accuracy: 0.9500
    Epoch 174/400
    1/1 [==============================] - 0s 29ms/step - loss: 0.2823 - binary_accuracy: 0.9100
    Epoch 175/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.2791 - binary_accuracy: 0.9500
    Epoch 176/400
    1/1 [==============================] - 0s 59ms/step - loss: 0.2759 - binary_accuracy: 0.9100
    Epoch 177/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.2726 - binary_accuracy: 0.9400
    Epoch 178/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.2695 - binary_accuracy: 0.9400
    Epoch 179/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.2675 - binary_accuracy: 0.9100
    Epoch 180/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.2663 - binary_accuracy: 0.9400
    Epoch 181/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2648 - binary_accuracy: 0.9200
    Epoch 182/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.2640 - binary_accuracy: 0.9700
    Epoch 183/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.2645 - binary_accuracy: 0.9100
    Epoch 184/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2642 - binary_accuracy: 0.9600
    Epoch 185/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.2647 - binary_accuracy: 0.8900
    Epoch 186/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2607 - binary_accuracy: 0.9600
    Epoch 187/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2569 - binary_accuracy: 0.9000
    Epoch 188/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.2520 - binary_accuracy: 0.9600
    Epoch 189/400
    1/1 [==============================] - 0s 24ms/step - loss: 0.2480 - binary_accuracy: 0.9300
    Epoch 190/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.2452 - binary_accuracy: 0.9500
    Epoch 191/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2437 - binary_accuracy: 0.9600
    Epoch 192/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.2429 - binary_accuracy: 0.9300
    Epoch 193/400
    1/1 [==============================] - 0s 74ms/step - loss: 0.2430 - binary_accuracy: 0.9900
    Epoch 194/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.2433 - binary_accuracy: 0.9200
    Epoch 195/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2422 - binary_accuracy: 0.9600
    Epoch 196/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.2423 - binary_accuracy: 0.9000
    Epoch 197/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.2426 - binary_accuracy: 0.9300
    Epoch 198/400
    1/1 [==============================] - 0s 30ms/step - loss: 0.2451 - binary_accuracy: 0.9000
    Epoch 199/400
    1/1 [==============================] - 0s 28ms/step - loss: 0.2409 - binary_accuracy: 0.9300
    Epoch 200/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.2348 - binary_accuracy: 0.9100
    Epoch 201/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2267 - binary_accuracy: 0.9800
    Epoch 202/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2227 - binary_accuracy: 0.9700
    Epoch 203/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.2228 - binary_accuracy: 0.9400
    Epoch 204/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.2241 - binary_accuracy: 0.9900
    Epoch 205/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.2248 - binary_accuracy: 0.9200
    Epoch 206/400
    1/1 [==============================] - 0s 38ms/step - loss: 0.2199 - binary_accuracy: 0.9600
    Epoch 207/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.2160 - binary_accuracy: 0.9500
    Epoch 208/400
    1/1 [==============================] - 0s 55ms/step - loss: 0.2125 - binary_accuracy: 0.9600
    Epoch 209/400
    1/1 [==============================] - 0s 38ms/step - loss: 0.2105 - binary_accuracy: 0.9500
    Epoch 210/400
    1/1 [==============================] - 0s 54ms/step - loss: 0.2077 - binary_accuracy: 0.9500
    Epoch 211/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.2042 - binary_accuracy: 0.9500
    Epoch 212/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.2021 - binary_accuracy: 0.9600
    Epoch 213/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.2011 - binary_accuracy: 0.9600
    Epoch 214/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1996 - binary_accuracy: 0.9600
    Epoch 215/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1981 - binary_accuracy: 0.9600
    Epoch 216/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1971 - binary_accuracy: 0.9600
    Epoch 217/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.2000 - binary_accuracy: 0.9300
    Epoch 218/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.2386 - binary_accuracy: 0.8700
    Epoch 219/400
    1/1 [==============================] - 0s 30ms/step - loss: 0.3887 - binary_accuracy: 0.7900
    Epoch 220/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.5775 - binary_accuracy: 0.7200
    Epoch 221/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.3597 - binary_accuracy: 0.7900
    Epoch 222/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.5996 - binary_accuracy: 0.6900
    Epoch 223/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1934 - binary_accuracy: 0.9500
    Epoch 224/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.4193 - binary_accuracy: 0.7700
    Epoch 225/400
    1/1 [==============================] - 0s 95ms/step - loss: 0.2461 - binary_accuracy: 0.8800
    Epoch 226/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.3256 - binary_accuracy: 0.8000
    Epoch 227/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.2605 - binary_accuracy: 0.8700
    Epoch 228/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.2426 - binary_accuracy: 0.8900
    Epoch 229/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.3086 - binary_accuracy: 0.8200
    Epoch 230/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1947 - binary_accuracy: 0.9600
    Epoch 231/400


    1/1 [==============================] - 0s 26ms/step - loss: 0.2859 - binary_accuracy: 0.8400
    Epoch 232/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.2181 - binary_accuracy: 0.9200
    Epoch 233/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.2250 - binary_accuracy: 0.9000
    Epoch 234/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.2598 - binary_accuracy: 0.8600
    Epoch 235/400
    1/1 [==============================] - 0s 24ms/step - loss: 0.1977 - binary_accuracy: 0.9500
    Epoch 236/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.2384 - binary_accuracy: 0.9000
    Epoch 237/400
    1/1 [==============================] - 0s 21ms/step - loss: 0.2121 - binary_accuracy: 0.9300
    Epoch 238/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.2026 - binary_accuracy: 0.9200
    Epoch 239/400
    1/1 [==============================] - 0s 42ms/step - loss: 0.2280 - binary_accuracy: 0.9000
    Epoch 240/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.1915 - binary_accuracy: 0.9500
    Epoch 241/400
    1/1 [==============================] - 0s 30ms/step - loss: 0.2138 - binary_accuracy: 0.9400
    Epoch 242/400
    1/1 [==============================] - 0s 73ms/step - loss: 0.2035 - binary_accuracy: 0.9400
    Epoch 243/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.1924 - binary_accuracy: 0.9400
    Epoch 244/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.2100 - binary_accuracy: 0.9000
    Epoch 245/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1882 - binary_accuracy: 0.9600
    Epoch 246/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1986 - binary_accuracy: 0.9400
    Epoch 247/400
    1/1 [==============================] - 0s 55ms/step - loss: 0.1951 - binary_accuracy: 0.9500
    Epoch 248/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1863 - binary_accuracy: 0.9500
    Epoch 249/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.1967 - binary_accuracy: 0.9200
    Epoch 250/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1833 - binary_accuracy: 0.9600
    Epoch 251/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1885 - binary_accuracy: 0.9600
    Epoch 252/400
    1/1 [==============================] - 0s 61ms/step - loss: 0.1858 - binary_accuracy: 0.9600
    Epoch 253/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1791 - binary_accuracy: 0.9500
    Epoch 254/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1852 - binary_accuracy: 0.9500
    Epoch 255/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1771 - binary_accuracy: 0.9600
    Epoch 256/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1820 - binary_accuracy: 0.9500
    Epoch 257/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1785 - binary_accuracy: 0.9500
    Epoch 258/400
    1/1 [==============================] - 0s 39ms/step - loss: 0.1776 - binary_accuracy: 0.9500
    Epoch 259/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1799 - binary_accuracy: 0.9600
    Epoch 260/400
    1/1 [==============================] - 0s 50ms/step - loss: 0.1749 - binary_accuracy: 0.9600
    Epoch 261/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1783 - binary_accuracy: 0.9500
    Epoch 262/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1743 - binary_accuracy: 0.9600
    Epoch 263/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.1753 - binary_accuracy: 0.9600
    Epoch 264/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1743 - binary_accuracy: 0.9600
    Epoch 265/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.1721 - binary_accuracy: 0.9600
    Epoch 266/400
    1/1 [==============================] - 0s 7ms/step - loss: 0.1735 - binary_accuracy: 0.9500
    Epoch 267/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1705 - binary_accuracy: 0.9600
    Epoch 268/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.1721 - binary_accuracy: 0.9500
    Epoch 269/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1698 - binary_accuracy: 0.9600
    Epoch 270/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.1702 - binary_accuracy: 0.9600
    Epoch 271/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1693 - binary_accuracy: 0.9600
    Epoch 272/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.1683 - binary_accuracy: 0.9500
    Epoch 273/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1687 - binary_accuracy: 0.9500
    Epoch 274/400
    1/1 [==============================] - 0s 40ms/step - loss: 0.1669 - binary_accuracy: 0.9600
    Epoch 275/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1672 - binary_accuracy: 0.9600
    Epoch 276/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1659 - binary_accuracy: 0.9700
    Epoch 277/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1655 - binary_accuracy: 0.9600
    Epoch 278/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1648 - binary_accuracy: 0.9600
    Epoch 279/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1638 - binary_accuracy: 0.9700
    Epoch 280/400
    1/1 [==============================] - 0s 43ms/step - loss: 0.1639 - binary_accuracy: 0.9600
    Epoch 281/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1624 - binary_accuracy: 0.9700
    Epoch 282/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1624 - binary_accuracy: 0.9700
    Epoch 283/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1611 - binary_accuracy: 0.9700
    Epoch 284/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1610 - binary_accuracy: 0.9700
    Epoch 285/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1599 - binary_accuracy: 0.9700
    Epoch 286/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1595 - binary_accuracy: 0.9700
    Epoch 287/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.1587 - binary_accuracy: 0.9700
    Epoch 288/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1581 - binary_accuracy: 0.9700
    Epoch 289/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1576 - binary_accuracy: 0.9700
    Epoch 290/400
    1/1 [==============================] - 0s 66ms/step - loss: 0.1569 - binary_accuracy: 0.9700
    Epoch 291/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1566 - binary_accuracy: 0.9700
    Epoch 292/400
    1/1 [==============================] - 0s 20ms/step - loss: 0.1559 - binary_accuracy: 0.9700
    Epoch 293/400
    1/1 [==============================] - 0s 36ms/step - loss: 0.1555 - binary_accuracy: 0.9700
    Epoch 294/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1549 - binary_accuracy: 0.9700
    Epoch 295/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1544 - binary_accuracy: 0.9700
    Epoch 296/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1537 - binary_accuracy: 0.9700
    Epoch 297/400
    1/1 [==============================] - 0s 52ms/step - loss: 0.1531 - binary_accuracy: 0.9700
    Epoch 298/400
    1/1 [==============================] - 0s 33ms/step - loss: 0.1526 - binary_accuracy: 0.9700
    Epoch 299/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.1522 - binary_accuracy: 0.9700
    Epoch 300/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1519 - binary_accuracy: 0.9700
    Epoch 301/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1515 - binary_accuracy: 0.9700
    Epoch 302/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1510 - binary_accuracy: 0.9700
    Epoch 303/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1503 - binary_accuracy: 0.9700
    Epoch 304/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1498 - binary_accuracy: 0.9700
    Epoch 305/400
    1/1 [==============================] - 0s 24ms/step - loss: 0.1493 - binary_accuracy: 0.9700
    Epoch 306/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1490 - binary_accuracy: 0.9700
    Epoch 307/400


    1/1 [==============================] - 0s 11ms/step - loss: 0.1485 - binary_accuracy: 0.9700
    Epoch 308/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1481 - binary_accuracy: 0.9700
    Epoch 309/400
    1/1 [==============================] - 0s 53ms/step - loss: 0.1475 - binary_accuracy: 0.9700
    Epoch 310/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1471 - binary_accuracy: 0.9700
    Epoch 311/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1466 - binary_accuracy: 0.9700
    Epoch 312/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1465 - binary_accuracy: 0.9700
    Epoch 313/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1457 - binary_accuracy: 0.9700
    Epoch 314/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1453 - binary_accuracy: 0.9700
    Epoch 315/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.1449 - binary_accuracy: 0.9700
    Epoch 316/400
    1/1 [==============================] - 0s 36ms/step - loss: 0.1446 - binary_accuracy: 0.9700
    Epoch 317/400
    1/1 [==============================] - 0s 49ms/step - loss: 0.1441 - binary_accuracy: 0.9700
    Epoch 318/400
    1/1 [==============================] - 0s 63ms/step - loss: 0.1437 - binary_accuracy: 0.9700
    Epoch 319/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1432 - binary_accuracy: 0.9700
    Epoch 320/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1428 - binary_accuracy: 0.9700
    Epoch 321/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.1424 - binary_accuracy: 0.9700
    Epoch 322/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1420 - binary_accuracy: 0.9700
    Epoch 323/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1415 - binary_accuracy: 0.9700
    Epoch 324/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1411 - binary_accuracy: 0.9700
    Epoch 325/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1407 - binary_accuracy: 0.9700
    Epoch 326/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1403 - binary_accuracy: 0.9700
    Epoch 327/400
    1/1 [==============================] - 0s 55ms/step - loss: 0.1398 - binary_accuracy: 0.9700
    Epoch 328/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1395 - binary_accuracy: 0.9700
    Epoch 329/400
    1/1 [==============================] - 0s 40ms/step - loss: 0.1391 - binary_accuracy: 0.9700
    Epoch 330/400
    1/1 [==============================] - 0s 47ms/step - loss: 0.1387 - binary_accuracy: 0.9700
    Epoch 331/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.1382 - binary_accuracy: 0.9700
    Epoch 332/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1380 - binary_accuracy: 0.9700
    Epoch 333/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1375 - binary_accuracy: 0.9700
    Epoch 334/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1371 - binary_accuracy: 0.9700
    Epoch 335/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1367 - binary_accuracy: 0.9700
    Epoch 336/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1364 - binary_accuracy: 0.9700
    Epoch 337/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1360 - binary_accuracy: 0.9700
    Epoch 338/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1356 - binary_accuracy: 0.9700
    Epoch 339/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1353 - binary_accuracy: 0.9700
    Epoch 340/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1349 - binary_accuracy: 0.9700
    Epoch 341/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.1345 - binary_accuracy: 0.9700
    Epoch 342/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1341 - binary_accuracy: 0.9700
    Epoch 343/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1338 - binary_accuracy: 0.9700
    Epoch 344/400
    1/1 [==============================] - 0s 38ms/step - loss: 0.1333 - binary_accuracy: 0.9700
    Epoch 345/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.1331 - binary_accuracy: 0.9700
    Epoch 346/400
    1/1 [==============================] - 0s 35ms/step - loss: 0.1326 - binary_accuracy: 0.9700
    Epoch 347/400
    1/1 [==============================] - 0s 57ms/step - loss: 0.1322 - binary_accuracy: 0.9700
    Epoch 348/400
    1/1 [==============================] - 0s 35ms/step - loss: 0.1320 - binary_accuracy: 0.9700
    Epoch 349/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1315 - binary_accuracy: 0.9700
    Epoch 350/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1312 - binary_accuracy: 0.9700
    Epoch 351/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1309 - binary_accuracy: 0.9700
    Epoch 352/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1304 - binary_accuracy: 0.9700
    Epoch 353/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1301 - binary_accuracy: 0.9700
    Epoch 354/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1297 - binary_accuracy: 0.9700
    Epoch 355/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1296 - binary_accuracy: 0.9700
    Epoch 356/400
    1/1 [==============================] - 0s 8ms/step - loss: 0.1290 - binary_accuracy: 0.9700
    Epoch 357/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1287 - binary_accuracy: 0.9700
    Epoch 358/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.1283 - binary_accuracy: 0.9700
    Epoch 359/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1280 - binary_accuracy: 0.9700
    Epoch 360/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1277 - binary_accuracy: 0.9700
    Epoch 361/400
    1/1 [==============================] - 0s 26ms/step - loss: 0.1274 - binary_accuracy: 0.9700
    Epoch 362/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1271 - binary_accuracy: 0.9700
    Epoch 363/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1268 - binary_accuracy: 0.9800
    Epoch 364/400
    1/1 [==============================] - 0s 27ms/step - loss: 0.1264 - binary_accuracy: 0.9700
    Epoch 365/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1261 - binary_accuracy: 0.9700
    Epoch 366/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1258 - binary_accuracy: 0.9800
    Epoch 367/400
    1/1 [==============================] - 0s 10ms/step - loss: 0.1254 - binary_accuracy: 0.9800
    Epoch 368/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1251 - binary_accuracy: 0.9800
    Epoch 369/400
    1/1 [==============================] - 0s 11ms/step - loss: 0.1249 - binary_accuracy: 0.9800
    Epoch 370/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1243 - binary_accuracy: 0.9800
    Epoch 371/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1240 - binary_accuracy: 0.9800
    Epoch 372/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1238 - binary_accuracy: 0.9800
    Epoch 373/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1233 - binary_accuracy: 0.9800
    Epoch 374/400
    1/1 [==============================] - 0s 31ms/step - loss: 0.1231 - binary_accuracy: 0.9800
    Epoch 375/400
    1/1 [==============================] - 0s 14ms/step - loss: 0.1229 - binary_accuracy: 0.9800
    Epoch 376/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.1223 - binary_accuracy: 0.9900
    Epoch 377/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.1220 - binary_accuracy: 0.9800
    Epoch 378/400
    1/1 [==============================] - 0s 15ms/step - loss: 0.1219 - binary_accuracy: 0.9800
    Epoch 379/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1214 - binary_accuracy: 0.9800
    Epoch 380/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1212 - binary_accuracy: 0.9800
    Epoch 381/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.1211 - binary_accuracy: 0.9800
    Epoch 382/400
    1/1 [==============================] - 0s 34ms/step - loss: 0.1204 - binary_accuracy: 0.9900
    Epoch 383/400


    1/1 [==============================] - 0s 17ms/step - loss: 0.1201 - binary_accuracy: 0.9900
    Epoch 384/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1199 - binary_accuracy: 0.9900
    Epoch 385/400
    1/1 [==============================] - 0s 23ms/step - loss: 0.1196 - binary_accuracy: 0.9900
    Epoch 386/400
    1/1 [==============================] - 0s 16ms/step - loss: 0.1193 - binary_accuracy: 0.9900
    Epoch 387/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1191 - binary_accuracy: 0.9900
    Epoch 388/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1189 - binary_accuracy: 0.9900
    Epoch 389/400
    1/1 [==============================] - 0s 19ms/step - loss: 0.1186 - binary_accuracy: 0.9900
    Epoch 390/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.1182 - binary_accuracy: 0.9900
    Epoch 391/400
    1/1 [==============================] - 0s 13ms/step - loss: 0.1180 - binary_accuracy: 0.9900
    Epoch 392/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1178 - binary_accuracy: 0.9800
    Epoch 393/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.1173 - binary_accuracy: 0.9900
    Epoch 394/400
    1/1 [==============================] - 0s 18ms/step - loss: 0.1169 - binary_accuracy: 0.9900
    Epoch 395/400
    1/1 [==============================] - 0s 17ms/step - loss: 0.1169 - binary_accuracy: 0.9900
    Epoch 396/400
    1/1 [==============================] - 0s 12ms/step - loss: 0.1165 - binary_accuracy: 0.9900
    Epoch 397/400
    1/1 [==============================] - 0s 46ms/step - loss: 0.1161 - binary_accuracy: 0.9900
    Epoch 398/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.1158 - binary_accuracy: 0.9900
    Epoch 399/400
    1/1 [==============================] - 0s 25ms/step - loss: 0.1156 - binary_accuracy: 0.9900
    Epoch 400/400
    1/1 [==============================] - 0s 9ms/step - loss: 0.1153 - binary_accuracy: 0.9900


model.fit() returns a History object that contains a dictionary with everything that happened during training. We can use this values to plot the loss and training accuracy with respect iterations.


```python
history_dict = history.history
history_dict.keys()
```


```python
acc = history_dict['binary_accuracy']
loss = history_dict['loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


```python
plt.plot(epochs, acc, 'r', label='Training acc')

plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```


```python

```


```python

```
