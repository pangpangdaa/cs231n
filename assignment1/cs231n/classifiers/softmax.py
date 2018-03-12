import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)
  maxLogC=np.max(scores,axis=1).reshape(-1,1)
  expScores=np.exp(scores+maxLogC)
  for i in range(num_train):
    esum=np.sum(expScores[i])
    eyi=expScores[i,y[i]]
    li=-np.log(eyi/esum)
    loss+=li
    dW[:,y[i]]-=X[i]
    for j in range(num_classes):
        dW[:,j]+=(expScores[i,j]/esum)*X[i]
  loss/=num_train
  dW/=num_train
  dW+=reg*W
  loss+=reg*np.sum(W*W)
   
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

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
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=np.exp(X.dot(W))
  scores=scores+np.max(scores,axis=1).reshape(-1,1)
  yi=scores[np.arange(num_train),y]
  loss+=np.sum(-np.log(yi/np.sum(scores,axis=1)))
  loss/=num_train
  loss+=reg*np.sum(W*W)
  sum=np.sum(scores,axis=1).reshape(-1,1)
  matrix=scores/sum
  matrix[np.arange(num_train),y]-=-1
  dW=X.T.dot(matrix)
  dW/=num_train
  dW+=reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

