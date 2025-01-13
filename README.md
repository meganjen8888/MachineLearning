# MachineLearning

Third programming assignment: We have so far learned how to implement a linear regression model for regression task and implement a logistic regression model for classification task. The essential steps in the implementations are:
1.	Define your hypothesis, for example hw(x)=wTx for linear regression model where both w and x are vectors;
2.	Define your cost or total loss function, and given the training data, calculate the cost or total loss comparing with the true labels or values of the training examples;
3.	Use gradient descent to optimize the cost function, and find the parameters in your hypothesis.
4.	Use the hypothesis with the found parameters to predict on unseen data.
And we have introduced machine learning library scikit-learn,which has many models already built for you, we can just import linear regression or logistic regression model and fit our training data to learn a model to predict on unseen data. But for deep learning, scikit-learn is not suitable anymore (non GPU optimizations), so we introduced you popular deep learning frameworks Pytorch and TensorFlow.
In this assignment, you will first see how to use scikit-learn to train logistic regression model for assignment 2 data (student admission), and then practice with Pytorch basics and then use Pytorch and TensorFlow to build a neural network model for a house price prediction (regression task) and a neural network model for predicting breast cancer (classification task).
To submit: You will need to submit this notebook with proper naming Assignment3_FirstName_LastName.ipynb to blackboard.
Important: Remember to restart the kernel everytime you modify your model or training steps Otherwise you may see some strange behavior which may be caused by "model" training being accumulated.

![image](https://github.com/user-attachments/assets/084360df-f08c-49f1-886a-b152b944c7e3)
