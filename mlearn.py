'''
Machine learning functions

Built for the Higgs classification challenge, but should be generalized
for other challenges as well.

Dan Morris
5/21/2014 -> 6/4/2014
'''

'''
Conventions:
  example feature matrix comes in 'X' (m x n)
  weights vector comes in 'w' (m x 1)
  outcomes vector comes in 'y' (m x 1)
  theta matrix comes in 'theta' (n x 1)

'''
import numpy as np
from random import random, shuffle
from matrixops import *
from base import *
#from sklearn import svm
import simplejson as json

''' Logistic Regression Functions'''
# sigmoid returns the sigmoid/logistic function for a scalar, vector, or matrix
def sigmoid(v):
  return 1/(1+np.exp(-v))
def h_logit(x,t):
  return sigmoid(x*t)

# pred_logit returns a prediction and confidence given h_theta(x) and threshold
def pred_logit(h,threshold=.5):
  if h >= threshold:
    return 1
    #c = (h - threshold) / (1 - threshold)
  else:
    return 0
    #c = (threshold - h) / threshold

# J_logistic is the cost function for logistic regression
# J_logistic_reg adds regularization
def J_logistic(X, y, theta):
  m,n = X.shape
  h = h_logit(X,theta)
  J = np.transpose(-y) * np.log(h) - np.transpose(1-y) * np.log(1-h)
  return float(J) / m
def J_logistic_reg(X,y,theta,lamb):
  m,n = X.shape
  h = h_logit(X,theta)
  J = np.transpose(-y) * np.log(h) - np.transpose(1-y) * np.log(1-h)
  J += lamb * np.transpose(theta)*theta
  return float(J) / m

# SGD takes a step of stochastic gradient descent and returns the new theta
# xi, yi is a single training example
# SGD_reg is regularized
def SGD(xi, yi, theta, alpha):
  h = h_logit(xi,theta)
  pd = np.transpose(xi) * (yi - h)
  return theta + alpha * pd
def SGD_reg(xi, yi, theta, alpha, lamb):
  h = h_logit(xi,theta)
  pd = np.transpose(xi) * (yi - h)
  theta[1:] *= (1-alpha * lamb)
  return theta + alpha * pd

# batch_GD takes a step of gradient descent with a batch of m examples
# batch_GD_reg is regularized
def batch_GD(xb, yb, theta, alpha):
  m,n = xb.shape
  h = h_logit(xb,theta)
  pd = (np.transpose(xb) * (yb - h)) / m
  return theta + alpha * pd
def batch_GD_reg(xb, yb, theta, alpha, lamb):
  m,n = xb.shape
  h = h_logit(xb,theta)
  pd = (np.transpose(xb) * (yb - h)) / m
  theta[1:] *= (1-alpha * lamb)
  return theta + alpha * pd

# logistic_accuracy returns tp, fp, tn, fn rates
def logistic_accuracy(X,y,theta,threshold=.5):
  m,n = X.shape
  h = sigmoid(X * theta)
  tp = 0.
  fp = 0.
  fn = 0.
  tn = 0.
  for i in range(m):
    if h[i] >= threshold:
      if int(y[i]) == 1:
        tp += 1
      else:
        fp += 1
    else:
      if int(y[i]) == 1:
        fn += 1
      else:
        tn += 1
  tp /= m
  fp /= m
  tn /= m
  fn /= m
  return [tp,fp,fn,tn]

def simple_train_logit(X,y,iters=100000,batch=50,alpha=.0003,theta=None,\
                      notify=10000,finalJ=False,plotJ=False,printiters=True):
  m,n = X.shape
  if plotJ:
    Jp = []
  if theta == None:
    theta = rand_init(n+1,1)
  X = add_zero_feature(X)
  for i in xrange(iters):
    if printiters and i%notify == 0:
      print 'Iteration ' + str(i) + ' of ' + str(iters)
    if plotJ:
      Jp.append(J_logistic(X,y,theta))
    bstart = (batch*i)%m
    bfinish = bstart+batch
    theta = batch_GD(X[bstart:bfinish,:], y[bstart:bfinish], theta, alpha)
  if finalJ:
    Jf = J_logistic(X,y,theta)
    return (theta, Jf)
  if plotJ:
    # TODO
    return theta
  else:
    return theta

def optimized_theta_STL(X,y,inits=10):
  # Runs simple_train_logit for small iterations with various
  # initial thetas and chooses the one which converges to lowest cost function
  Jbest = 100000 # stupid high for init, first example will def be lower
  m,n = X.shape
  for t in range(inits):
    ti = rand_init(n+1,1,.3)
    tf,J = simple_train_logit(X,y,iters=5000,theta=ti,finalJ=True,plotJ=False,\
                             printiters=False)
    print 'J '+str(n) + ' = ' + str(J)
    if J < Jbest:
      Jbest = J
      theta = tf
  # Train a high-iters model with the best current theta
  return simple_train_logit(X,y,iters=200000,theta=theta,finalJ=False,\
                           plotJ=False,printiters=False)
''' -----------------------------'''

''' Neural Network Functions '''
class NN:
  '''---Variable descriptions and sizes---
    Input Variables
      X: the feature matrix (m x n matrix)
      y: the outcome matrix (m x 1 matrix)
      l: the # of hidden layers (int)
      f: the # hidden features (int)
      fileroot: the base for a filename to save parameters. Should be unique
        to the data set imported.
    Created Variables
      mtotal: the total number of examples in the input feature matrix (int)
      n: the number of features in each input example xi (int)
      m+train/cv/test: the number of examples in the given set (int)
      X+train/cv/test: the feature matrix of the given set (m+ x n matrix)
      y+train/cv/test: the outcome matrix of the given set (m+ x 1 matrix)
      h+train/cv/test: the prediction matrix of the given set (m+ x 1 matrix)
      hiddenlayers: the number of hidden layers (int)
      L: the index of the outcome layer (int)
      hiddenfeatures: the number of hidden features (int)
      bias: a vector of ones to add to layers (mtrain x 1 matrix)
      Theta[l]: parameter matrix between layers l and l+1 (matrix)
      A[l]: Activations of layer l, including input and output layers (matrix)
      a[l]: temporary activation layers for a single example xi (1 x n,f,1 matrix)
      Delta[l]: partial derivatives for Theta[l] (matrix)
      delta[l]: error components for layer l (1 x f,1 matrix)
      Jhistory: the J values for each iteration of training (list)
      ----------------------------------------'''
  def __init__(self, l, f, fileroot):
    print 'Initializing Neural Network: '+fileroot
    self.hiddenlayers = l
    self.fileroot = fileroot
    self.L = l+1 # index of outcome/prediction layer
    self.hiddenfeatures = f
    return
  def training_setup(self,X,y):
    print 'Setting up Neural Network for training...'
    self.X = X
    self.y = y
    self.mtotal, self.n = X.shape
    self.shuffle_examples()
    self.split_training()
    self.A = [0]*(self.hiddenlayers+2)
    self.bias = np.ones((self.mtrain,1))
    self.A[0] = np.append(self.bias,self.Xtrain,axis=1)
    self.Delta = [0]*(self.hiddenlayers+1)
    self.layersizes = [self.n] # list of layer sizes
    for j in range(self.hiddenlayers):
      self.layersizes.append(self.hiddenfeatures)
    self.layersizes.append(1)
    thetas_loaded = self.loadThetas()
    if not thetas_loaded:
      print 'Randomizing Thetas...'
      self.Theta = []
      for i in range(self.hiddenlayers+1):
        self.Theta.append(rand_init(self.layersizes[i]+1,self.layersizes[i+1]))
    else:
      print 'Thetas loaded successfully!'
    '''print "layersize list: ", self.layersizes
    for t in range(self.hiddenlayers+1):
      print "Theta " + str(t) + " shape"
      print self.Theta[t].shape'''
    return
  def prediction_setup(self):
    thetas_loaded = self.loadThetas()
    if thetas_loaded:
      print 'Thetas loaded. Ready to predict.'
    else:
      print 'Failed to load Thetas. Check fileroot input.'
    return
  def shuffle_examples(self):
    # shuffles training examples randomly
    print 'Shuffling training examples...'
    T = np.append(self.X,self.y,axis=1)
    np.random.shuffle(T)
    self.X = T[:,:self.n]
    self.y = T[:,self.n]
    return
  def split_training(self, ratio = [.7,.2,.1]):
    print 'Splitting training set into train/cv/test...'
    self.mtrain = int(self.mtotal * ratio[0])
    self.mcv = int(self.mtotal * ratio[1])
    self.mtest = self.mtotal - self.mtrain - self.mcv
    # splits training set X,y into training, cross-validation, testing sets
    self.Xtrain = self.X[:self.mtrain,:]
    self.ytrain = self.y[:self.mtrain]
    self.Xcv = self.X[self.mtrain:self.mtrain+self.mcv,:]
    self.ycv = self.y[self.mtrain:self.mtrain+self.mcv]
    self.Xtest = self.X[self.mtrain+self.mcv:]
    self.ytest = self.y[self.mtrain+self.mcv:]
    # initialize h (hypotheses) matrices for each set
    self.htrain = np.ones(self.ytrain.shape)
    self.hcv = np.ones(self.ycv.shape)
    self.htest = np.ones(self.ytest.shape)
    return
  def J(self,s='train'):
    # Cost function of the network with current parameters Theta on set 's'
    # Note: Cannot currently handle regularization
    if s == 'train':
      j = np.transpose(-self.ytrain) * np.log(self.htrain)
      j -= np.transpose(1-self.ytrain) * np.log(1-self.htrain)
      j = j / self.mtrain
    elif s == 'cv':
      j = np.transpose(-self.ycv) * np.log(self.hcv)
      j -= np.transpose(1-self.ycv) * np.log(1-self.hcv)
      j = j / self.mcv
    elif s == 'test':
      j = np.transpose(-self.ytest) * np.log(self.htest)
      j -= np.transpose(1-self.ytest) * np.log(1-self.htest)
      j = j / self.mtest
    return float(j)
  def train(self,iters,alpha=.1,Jdensity=50,verbose=False):
    print 'Training Neural Network...'
    self.Jtrain = []
    self.Jcv = []
    for i in xrange(iters):
      # forward propagate to set hypotheses and activation layers
      self.forprop()
      # initialize empty Deltas
      for l in range(self.hiddenlayers+1):
        self.Delta[l] = np.mat(np.zeros(self.Theta[l].shape))
      # iterate over all training examples
      for j in xrange(self.mtrain):
        # backwards propagate each example, adding its error to Delta
        self.backprop(j)
      for l in range(self.L):
        self.Delta[l] = self.Delta[l] / self.mtrain # equalize for m
      self.graddesc(alpha)
      if i%Jdensity == 0:
        self.Jtrain.append(self.J('train'))
        self.checkcv()
        self.Jcv.append(self.J('cv'))
        if verbose:
          print 'iter ' + str(i) + ' Jtrain: ' + str(self.Jtrain[i/Jdensity])\
              + '  Jcv:' + str(self.Jcv[i/Jdensity])
    if not verbose:
      print 'Initial J: ' + str(self.Jtrain[0])
      print 'Final J: ' + str(self.Jtrain[-1])
    return
  def checkcv(self):
    # Computes hcv for the cross-validation set.
    for i in xrange(self.mcv):
      self.hcv[i] = self.predict(self.Xcv[i,:])
    return
  def forprop(self):
    # forward propagation for all training examples
    for l in range(self.hiddenlayers+1):
      # push layer l through Theta[l] to get layer l+1, add bias unit if not last
      self.A[l+1] = sigmoid(self.A[l]*self.Theta[l])
      if l < self.hiddenlayers:
        self.A[l+1] = np.append(self.bias,self.A[l+1],axis=1)
    self.htrain = self.A[self.hiddenlayers+1]
    return
  def backprop(self,j):
    # initialize a layers (1 x n, 1 x f..., 1 x 1)
    a = []
    for l in range(self.L+1):
      a.append(self.A[l][j])
    # initialize empty deltas
    delta = []
    for l in range(self.L):
      delta.append(np.mat(np.zeros(a[l+1].shape)))
    # calculate deltas for each layer
    delta[self.L-1] = a[self.L] - self.y[j] # output layer delta
    for l in range(self.L-1,0,-1): # each hidden layer, backwards
      atemp = np.array(a[l])
      sigprime = atemp*(1-atemp) # sigprime = array, derivative of sigmoid
      if self.L - l > 1: # multiple hidden layers
        delta[l-1] = np.mat(np.array(delta[l][0,1:] * np.transpose(self.Theta[l])) * sigprime)
      else:
        delta[l-1] = np.mat(np.array(delta[l] * np.transpose(self.Theta[l])) * sigprime)
    # add this example's errors to Deltas
    for l in range(self.L):
      if l == self.L-1:
        self.Delta[l] += np.transpose(a[l])*delta[l]
      else:
        self.Delta[l] += np.transpose(a[l])*(delta[l][0,1:]) #ignore bias units
    return
  def graddesc(self,alpha):
    # Takes a step of gradient descent, adjusting the Thetas
    for l in range(self.L):
      self.Theta[l] -= alpha * self.Delta[l]
    return
  def singlebias(self,x):
    # adds a bias unit to the beginning of a 1xn matrix
    return np.append(np.matrix('1'),x,axis=1)
  def predict(self,x):
    # predicts hypothesis h for an input example x with current parameters
    for l in range(self.L):
      # add bias unit
      x = self.singlebias(x)
      # push through Theta[l]
      x = sigmoid(x * self.Theta[l])
    return float(x)
  def thetanamebase(self):
    return 'NNData/'+self.fileroot + '_hl' + str(self.hiddenlayers) + '_hf' + \
           str(self.hiddenfeatures) + '.json'
  def saveThetas(self):
    fname = self.thetanamebase()
    t = {}
    for i in range(self.L):
      r,c = self.Theta[i].shape
      key = str(i)+'_'+str(r)+'_'+str(c)
      tlist = np.array(self.Theta[i]).ravel().tolist()
      t[key] = tlist
    with open(fname,'w') as wf:
      json.dump(t,wf)
    return
  def loadThetas(self):
    fname = self.thetanamebase()
    try:
      with open(fname) as rf:
        t = json.loads(rf.read())
        l = len(t)
        self.Theta = [0]*(l)
        for k in t:
          key = k.split('_')
          i = int(key[0])
          r = int(key[1])
          c = int(key[2])
          self.Theta[i] = np.mat(t[k]).reshape(r,c)
      return True
    except:
      return False
''' ------------------------ '''
