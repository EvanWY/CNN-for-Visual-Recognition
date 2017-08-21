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

  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  test_size = X.shape[0]

  for i in xrange(test_size):
    scores = X[i].dot(W)

    exp_scores = np.exp(scores)
    norm_exp_scores = exp_scores / np.sum(exp_scores)

    d_norm_exp_scores = -1.0 / norm_exp_scores[y[i]]

    d_exp_score = np.zeros_like(exp_scores)
    d_exp_score[:] = -(np.sum(exp_scores) ** -2) * exp_scores[y[i]] * d_norm_exp_scores
    d_exp_score[y[i]] = d_exp_score[y[i]] / exp_scores[y[i]] * (np.sum(exp_scores) - exp_scores[y[i]])
    
    d_score = exp_scores * d_exp_score
    d_score[y[i]] *= -1.0

    loss_i = -np.log(norm_exp_scores[y[i]])
    loss += loss_i

    dW += X[i].reshape(-1,1).dot(d_score.reshape(1,-1))

  loss /= test_size
  dW /= test_size

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #

  scores = X.dot(W)
  exp_scores = np.exp(scores)
  
  sum_exp_scores = np.sum(exp_scores, axis = 1)
  correct_exp_scores = exp_scores[xrange(exp_scores.shape[0]), y]

  probabilities = correct_exp_scores / sum_exp_scores

  loss_array = - np.log(probabilities)

  loss = np.average(loss_array) + 0.5 * reg * np.sum(W * W)

  d_loss_array = np.zeros_like(loss_array) + (1.0 / loss_array.shape[0])
  d_probabilities = (- 1.0 / probabilities) * d_loss_array

  d_correct_exp_scores = d_probabilities / sum_exp_scores
  d_sum_exp_scores = - correct_exp_scores * (sum_exp_scores ** -2) * d_probabilities

  d_exp_scores = np.zeros_like(exp_scores)
  d_exp_scores += d_sum_exp_scores.reshape(-1,1)
  d_exp_scores[xrange(exp_scores.shape[0]), y] += d_correct_exp_scores

  d_scores = d_exp_scores * exp_scores
  dW = X.T.dot(d_scores) + reg * W

  # print d_loss_array
  # print d_probabilities
  # print d_correct_exp_scores
  # print d_sum_exp_scores
  # print d_exp_scores
  # print d_scores
  # print dW

  return loss, dW

