'''
Matrix operations for the Higgs classification dataset.

Slice and dice.

Dan Morris
5/20/2014 ->
'''

import numpy as np
from random import random
import simplejson as json

# rand_init returns an r x c matrix of random values within (-eps,eps)
def rand_init(r,c, eps=.1):
  m = np.zeros(r*c)
  for i in range(r*c):
    m[i] = (random()*2 - 1) * eps
  m.shape = (r,c)
  return np.mat(m)

# get_mu_sigma returns the mean and SD of the columns of X
def get_mu_sigma(X):
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  return mu, sigma

# normalize subtracts mu and divides by sigma for each element in example
# normalize_all runs normalize on each row in X
def normalize(data, mu, sigma):
  data -= mu
  data = data / sigma
  return data
def normalize_all(X, mu, sigma):
  m,n = X.shape
  for row in xrange(m):
    X[row,:] = normalize(X[row,:], mu, sigma)
  return

# load_normalizers gets data from normalizers.json and returns a dict of
# mu and sigma
def load_normalizers():
  with open('norms.json') as rf:
    d = json.loads(rf.read())
    for k in d:
      d[k]['mu'] = np.array(d[k]['mu'])
      d[k]['sigma'] = np.array(d[k]['sigma'])
  return d

# load_thetas gets data from thetas.json and returns a dict of theta arrays
def load_thetas():
  with open('thetas.json') as rf:
    d = json.loads(rf.read())
    for k in d:
      d[k] = np.array(d[k])
  return d

# add_zero_feature adds a column of ones to the front of matrix X
def add_zero_feature(X):
  X = np.insert(X,0,1,axis=1)
  return X

# size_string returns a simple string of a matrix's size for testing purposes
def size_string(m):
  r,c = m.shape
  return str(r) + ' by ' + str(c)

def row_mat_to_list(row):
  r,c = row.shape
  l = []
  if r > 1:
    print 'Not a row.'
    return
  for i in range(c):
    l.append(row[0,i])
  return l
def list_to_row_mat(l):
  return np.mat(l)
