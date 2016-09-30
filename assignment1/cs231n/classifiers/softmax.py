import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
      scores = X[i].dot(W)
      # use the trick metioned in lecture for numeric stability
      scores -= np.max(scores)
      probability = np.exp(scores) / np.sum(np.exp(scores))
      loss += -np.log(probability[y[i]])
      # gradient for softmax loss
      # http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
      for j in xrange(num_classes):
          dW[:, j] += (probability[j] - (j == y[i])) * X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_train = X.shape[0]

  scores = X.dot(W)
  # use the trick metioned in lecture for numeric stability
  # reshape for brocasting
  max_score = np.amax(scores, axis=1).reshape(-1, 1)
  scores -= max_score
  # cache exp_scores because it is used twice and doing exponential element-wise could be expansive
  exp_scores = np.exp(scores)
  # reshape for brocasting
  probability = exp_scores / np.sum(exp_scores, axis=1).reshape(-1, 1)
  # access the probability of correct class y first, then sum the vector
  loss = -np.log(probability[np.arange(num_train), y]).sum()

  probability[np.arange(num_train), y] -= 1
  dW = X.T.dot(probability)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
