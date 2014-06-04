'''
Base python code for the Higgs Boson classifier problem

Contains functions specific to the Higgs challenge

Dan Morris
5/20/2014 -> 6/4/2014
'''

'''
Notes to self:
in Label, 's' = signal (1) and 'b' = background (0)
PRI (primitive) = raw data
DER (derived) = computed from PRI features by physicists

Conventions:
  Data files should be headless except for the original test.csv and train.csv
  Data csvs:
    col 0: example id. Useful only to assure that data is shuffled properly
    cols 1-30: features data
    col 31: weight. Useful for gauging accuracy by the AMS metric
    col 32: outcome. 'b' or 's'.

'''

import numpy as np
from random import shuffle, random, choice
import simplejson as json
from mlearn import *
from matrixops import *
from plot import *
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC

# Constants
train_total_count = 250000
train_complete_count = 68114 # no -999.0 terms
tc_train_count = 40137 # examples in tc-train.csv
tc_cv_count = 13591    # examples in tc-cv.csv
tc_test_count = 13486  # examples in tc-test.csv
train_incomplete_count = 181886 # includes at least one -999.0 term
ti_train_count = 108867 # examples in ti-test.csv
ti_cv_count = 36357     # examples in ti-cv.csv
ti_test_count = 36662   # examples in ti-test.csv
nfeatures = 30 # number of features in the data sets
jet_bins = ['0','0u','1','1u','2','2u','3','3u']
jet_bin_counts = {'0': 73790, '0u': 26123, '1': 69982, '1u': 7562, \
                  '2': 47427, '2u': 2952,  '3': 20687, '3u': 1477}
jet_bin_samples = {'0': 20000, '0u': 10000, '1': 20000, '1u': 5000, \
                   '2': 20000, '2u': 2952,  '3': 10000, '3u': 1477}
acc_dict = {'tp':1,'fp':2,'fn':3,'tn':4}
#bin_nfeatures = {'0': , '0u': , '1': , '1u': , \
#                 '2': , '2u': ,  '3': , '3u': }}

def load_weights(filename):
  with open('weights.json') as rf:
    d = json.loads(rf.read())
    try:
      return d[filename]
    except:
      print "Can't find weight data for file " + filename
      return None

# line_parse takes a line from a training or test file, converts it
# appropriately, and returns the relevant info in proper formats.
def line_parse(line, datatype='train'):
  line = line.strip('/n')
  line = line.split(',')
  lineid = line[0]
  u = ''
  if line[1]=='-999.0':
    u = 'u'
  jets = line[23] + u
  for i in range(1,31):
    line[i] = float(line[i])
  if datatype == 'train': # training data
    weight = float(line[31])
    if 's' in line[32]:
      outcome = 1
    else:
      outcome = 0
  else: # test data
    weight = None
    outcome = None
  return (lineid, jets, np.array(line[1:31]), weight, outcome)

# jet_line_parse is similar to line_parse but adjusts based on the # of jets
# and whether or not the first feature is defined. Sorts into 8 bins.
def jet_line_parse(line):
  data = []
  line = line.strip('/n')
  line = line.split(',')
  u = ''
  if line[1]=='-999.0':
    u = 'u'
  jets = line[23] + u # one of '0,0u,1,1u,2,2u,3,3u'
  lineid = line[0]
  for i in range(1,31):
    line[i] = float(line[i])
  if jets[0] == '0':
    data += line[1:5]
    data += line[8:13]
    data += line[14:23]
  elif jets[0] == '1':
    data += line[1:5]
    data += line[8:13]
    data += line[14:23]
    data += line[24:27]
  else:
    data += line[1:23]
    data += line[24:31]
  if u == 'u': # first feature is -999.0: remove it!
    data = data[1:]
  if len(line) > 31:
    weight = float(line[31])
    if 's' in line[32]:
      outcome = 1
    else:
      outcome = 0
  else: # test data
    weight = None
    outcome = None
  return (lineid, jets, np.array(data), weight, outcome)

# load_jetsplit_data creates matrices+vectors for modeling
# features (X) = m x 30 matrix of features
# weights (w)  = m x 1 vector of weights
# outcomes (y) = m x 1 vector of outcomes
def load_jetsplit_data(fname, m):
  with open(fname) as rf:
    lineid, jets, data, weight, outcome = jet_line_parse(rf.readline())
    n = len(data)
    X = np.mat(np.zeros((m,n)))
    w = np.mat(np.zeros((m,1)))
    y = np.mat(np.zeros((m,1)))
    X[0] = data
    w[0] = weight
    y[0] = outcome
    for i in xrange(1,m-1):
      lineid, jets, data, weight, outcome = jet_line_parse(rf.readline())
      X[i] = data
      w[i] = weight
      y[i] = outcome
  return X,w,y
def simple_load_data(fname, m, header=False):
  with open(fname) as rf:
    if header:
      rf.readline()
    X = np.mat(np.zeros((m,30)))
    w = np.mat(np.zeros((m,1)))
    y = np.mat(np.zeros((m,1)))
    for i in xrange(0,m):
      lineid, jets, data, weight, outcome = line_parse(rf.readline())
      X[i] = data
      w[i] = weight
      y[i] = outcome
  return X,w,y

# initialize_model compiles several functions to initialize modeling
# It returns a normalized feature matrix and weight + outcome vectors
def initialize_model(fname, m):
  X,w,y = load_jetsplit_data(fname, m)
  mu, sigma = get_mu_sigma(X)
  normalize_all(X,mu,sigma)
  return X,w,y

def simple_model_with_batchGD(trainfile, train_m, cvfile, cv_m, iters):
  X,w,y = initialize_model(trainfile, train_m)
  Xcv, wcv, ycv = initialize_model(cvfile, cv_m)
  m,n = X.shape
  theta = rand_init(m+1,1)
  X = add_zero_feature(X)
  Xcv = add_zero_feature(Xcv)
  J_train = []
  #J_cv = []
  '''tp_t = []
  fp_t = []
  fn_t = []
  tn_t = []
  tp_c = []
  fp_c = []
  fn_c = []
  tn_c = []'''
  alpha = .01
  lamb = .1
  for i in range(iters):
    if i%1000 == 0:
      print 'Iteration ' + str(i) + ' of ' + str(iters)
    J_train.append(J_logistic(X,y,theta))
    #J_cv.append(J_logistic(Xcv,ycv,theta))
    '''
    if i%500 == 0:
      train_acc = logistic_accuracy(X,y,theta)
      cv_acc = logistic_accuracy(Xcv,ycv,theta)
      tp_t.append(train_acc[0])
      fp_t.append(train_acc[1])
      fn_t.append(train_acc[2])
      tn_t.append(train_acc[3])
      tp_c.append(cv_acc[0])
      fp_c.append(cv_acc[1])
      fn_c.append(cv_acc[2])
      tn_c.append(cv_acc[3])
      '''
    theta = batch_GD_reg(X[i:i+10,:], y[i:i+10], theta, alpha, lamb)
  j_plot(J_smoother(J_train,10))
  #acc_plot_with_cv(tp_t,fp_t,fn_t,tn_t,tp_c,fp_c,fn_c,tn_c)
  return theta

# determine_alpha runs gradient descent with various alphas to find the best one
def determine_alpha(trainfile, cvfile, iters=50000):
  batch_size = 50
  train_size = 5000
  alpha_options = [.0003]
  for a in alpha_options:
    X,w,y = initialize_model(trainfile, train_size)
    Xcv, wcv, ycv = initialize_model(cvfile, 1000)
    theta = rand_init(nfeatures+1,1)
    X = add_zero_feature(X)
    Xcv = add_zero_feature(Xcv)
    J_train = []
    for i in xrange(iters):
      if i%1000 == 0:
        print 'Iteration ' + str(i) + ' of ' + str(iters)
      J_train.append(J_logistic(X,y,theta))
      bstart = i%train_size
      bfinish = i%train_size + batch_size
      theta = batch_GD(X[bstart:bfinish,:], y[bstart:bfinish], theta, a)
    j_plot(J_train)
  return theta

def simple_logit(trainfile, cvfile, m=1000, iters=50000):
  batch_size = 50
  train_size = m
  alpha = .0003
  X,w,y = initialize_model(trainfile, train_size)
  Xcv, wcv, ycv = initialize_model(cvfile, train_size)
  m,n = X.shape
  theta = rand_init(n+1,1)
  X = add_zero_feature(X)
  Xcv = add_zero_feature(Xcv)
  #J_train = []
  for i in xrange(iters):
    if i%1000 == 0:
      print 'Iteration ' + str(i) + ' of ' + str(iters)
    #J_train.append(J_logistic(X,y,theta))
    bstart = i%train_size
    bfinish = i%train_size + batch_size
    theta = batch_GD(X[bstart:bfinish,:], y[bstart:bfinish], theta, alpha)
  #j_plot(J_train)
  Jtr = J_logistic(X,y,theta)
  Jcv = J_logistic(Xcv,ycv,theta)
  return theta, Jtr, Jcv

# learning_curves models with various example sizes and plots them
# mrange is an array of example sizes m
def learning_curves(trainfile, cvfile, mrange, iters=20000):
  J_train = []
  J_cv = []
  for m in m_counts:
    theta, Jtr, Jcv = simple_logit(trainfile, cvfile, m, iters)
    J_train.append(Jtr)
    J_cv.append(Jcv)
  # plot it!
  plot_learning_curves(m_counts, J_train, J_cv)
  return

# predictor is the single-example prediction function to run on test sets
def predict_example_logit(testexample, normalizers, thetas, threshold=.5):
  # parse the line, set aside exampleid
  exampleid, jets, data, weight, outcome = jet_line_parse(testexample)
  mu = normalizers[jets]['mu']
  sigma = normalizers[jets]['sigma']
  data = normalize(data, mu, sigma)
  data = np.append([1],data)
  t = thetas[jets]
  h = sigmoid(np.dot(data, t))
  pred = pred_logit(h,threshold)
  return exampleid, h, pred
# predict_all predicts all examples in testfile. If traintest, true outcomes
# are available and an accuracy matrix will be produced.
def predict_all_logit(testfile, predfile, traintest=False, threshold=.5, \
                      header=False):
  n = load_normalizers()
  t = load_thetas()
  with open(testfile) as rf:
    if header:
      rf.readline()
    with open(predfile, 'w') as wf:
      while True:
        l = rf.readline()
        if l == '':
          break
        exampleid, confidence, pred = predict_example_logit(l,n,t,threshold)
        if traintest:
          if 's' in l:
            wf.write(exampleid+','+str(confidence)+','+pred+',s\n')
          else:
            wf.write(exampleid+','+str(confidence)+','+pred+',b\n')
        else:
          wf.write(exampleid+','+str(confidence)+','+pred+'\n')
  return

def two_feature_random_plotter(filename, previous, m=200, header=False):
  r = [2,3,4,8,9,10,11,14,15,16,17,18,19,20,21,22]
  a = choice(r)
  b = a
  while b == a:
    b = choice(r)
  while str(a)+'_'+str(b) in previous or str(b)+'_'+str(a) in previous:
    a = choice(r)
    b = a
    while b == a:
      b = choice(r)
  X,w,y = simple_load_data(filename,m,header)
  f1 = X[:,a]
  f2 = X[:,b]
  f1b = []
  f1s = []
  f2b = []
  f2s = []
  for i in xrange(m):
    if y[i] == 1:
      f1s.append(float(f1[i]))
      f2s.append(float(f2[i]))
    else:
      f1b.append(float(f1[i]))
      f2b.append(float(f2[i]))
  two_features_plot(f1b,f2b,f1s,f2s,str(a),str(b))
  return str(a)+'_'+str(b)

# histo_features plots a histogram of all features (or each in feature_array)
# specify which column has outcome(s) in outcome_col
def histo_features(fname,m,header=False,outcome_col=32,feature_array=None):
  features = np.zeros((m,30))
  with open(fname) as rf:
    if header:
      rf.readline()
    # load fname into a big array
    for i in xrange(m):
      l = rf.readline()
      if l=='':
        break
        # chop off the zeros from the bottom rows of the array?
      l = l.split(',')
      features[i] = l[:30]
  if feature_array != None:
    for f in feature_array:
      simple_histo(features[:,f])
  else:
    for f in range(len(features[1,:])):
      simple_histo(features[:,f])
  return

def test_predict_NN_8bins(testfile,predfile,traintest=False,header=False):
  # uses neural networks to predict each example in the test set
  nn = {}
  norms = load_normalizers()
  for k in jet_bins:
    # set up the NN for each bin
    nn[k] = NN(1,60,'jet'+k)
    nn[k].prediction_setup()
  # load test data, sending each example to the appropriate NN for prediction
  hrecord = []
  with open(testfile) as rf:
    with open(predfile,'w') as wf:
      if header:
        rf.readline()
      while True:
        l = rf.readline()
        if l == '':
          break
        lineid,jets,data,w,o = jet_line_parse(l)
        # normalize it!
        mu = norms[jets]['mu']
        sigma = norms[jets]['sigma']
        data = normalize(data, mu, sigma)
        h = nn[jets].predict(np.mat(data))
        hrecord.append(h)
        if h > .5:
          pred = 's'
        else:
          pred = 'b'
        if traintest:
          if 's' in l:
            wf.write(lineid+','+str(h)+','+pred+',s\n')
          else:
            wf.write(lineid+','+str(h)+','+pred+',b\n')
        else:
          wf.write(lineid+','+str(h)+','+pred+'\n')
  # process prediction files
  if not traintest:
    order_predictions(predfile, predfile)
  return hrecord

def nn_perpetual(l,f,iters=1000,Jinterval=100,verbose=True):
  while True:
    for k in jet_bins:
      X,w,y = initialize_model('Data/all_'+k+'.csv',jet_bin_samples[k])
      nn = NN(l,f,'jet'+k)
      nn.training_setup(X,y)
      nn.train(iters,.3,Jinterval,verbose)
      nn.saveThetas()
  return

def rf_8bins(fsize):
  # Trains a random forest with fsize trees for each jet bin
  # Returns a dict of rfc's keyed by jet bin
  rf_by_bin = {}
  for k in jet_bins:
    Xtr,Xcv,Xte,ytr,ycv,yte = rf_trainprep('data/all_'+k+'.csv',False)
    rf = rf_init(fsize)
    rf.fit(Xtr,ytr)
    rf_by_bin[k] = rf
  return rf_by_bin
def rf_pred_test(rf,x,y):
  # Predicts a single example from a training set, returns the outcome class
  p = rf.predict(x)
  if p == y and y == 1:
    return 'tp'
  elif p != y and y == 0:
    return 'fp'
  elif p != y and y == 1:
    return 'fn'
  elif p == y and y == 0:
    return 'tn'
  else:
    print 'Something went wrong.'
    return ''
def rf_8bin_accuracy(rf_dict,trainfile='training.csv',header=True):
  # Determines the accuracy of an 8bin rfc, returns an array for plotting
  acc = [0,0,0,0,0,0,0,0] # accuracy,tp,fp,fn,tn,precision,recall,F1
  with open(trainfile) as rf:
    if header:
      rf.readline()
    while True:
      l = rf.readline()
      if l == '':
        break
      lineid,jetbin,data,weight,y = line_parse(l)
      p = rf_pred_test(rf_dict[jetbin],data,y)
      acc[acc_dict[p]] += 1
  c = float(sum(acc))
  acc[0] = acc[1]+acc[4]
  for i in range(5):
    acc[i] = acc[i] / c
  acc[5] = float(acc[1]) / (acc[1]+acc[2])
  acc[6] = float(acc[1]) / (acc[1]+acc[3])
  acc[7] = 2*acc[5]*acc[6] / (acc[5] + acc[6])
  return acc
def how_many_trees(numseq=[10,25,60,100,250,600,1000]):
  # Calculates and plots accuracy of 8bin rfc with a given number of trees
  for n in numseq:
    rfd = rf_8bins(n)
    acc = rf_8bin_accuracy(rfd)
    print acc

''' Random Forest Functions '''
def rf_trainprep(trainfile='training.csv', header=True, weights=False):
  # Splits data from trainfile into train, cv, test for use in RFC
  # Returns X,y for each set
  # TODO: adjust for any number of input features
  train = []
  cv = []
  test = []
  with open(trainfile) as rf:
    if header:
      rf.readline()
    while True:
      l = rf.readline()
      if l == '':
        break
      # Assign to a set
      r = random()
      if r < .6:
        train.append(l)
      elif r < .8:
        cv.append(l)
      else:
        test.append(l)
    shuffle(train)
    shuffle(cv)
    shuffle(test)
    '''print len(train)
    print len(cv)
    print len(test)'''
    Xtrain = np.zeros((len(train),30))
    Xcv = np.zeros((len(cv),30))
    Xtest = np.zeros((len(test),30))
    ytrain = np.zeros(len(train))
    ycv = np.zeros(len(cv))
    ytest = np.zeros(len(test))
    if weights:
      wtrain = np.zeros(len(train))
      wcv = np.zeros(len(cv))
      wtest = np.zeros(len(test))
    '''print Xtrain.shape
    print Xcv.shape
    print Xtest.shape'''
    for i in range(len(train)):
      lineid,jets,data,weight,outcome = line_parse(train[i])
      Xtrain[i,:] = data
      ytrain[i] = outcome
      if weights:
        wtrain[i] = weight
    for i in range(len(cv)):
      lineid,jets,data,weight,outcome = line_parse(cv[i])
      Xcv[i,:] = data
      ycv[i] = outcome
      if weights:
        wcv[i] = weight
    for i in range(len(test)):
      lineid,jets,data,weight,outcome = line_parse(test[i])
      Xtest[i,:] = data
      ytest[i] = outcome
      if weights:
        wtest[i] = weight
  if weights:
    return Xtrain,Xcv,Xtest,ytrain,ycv,ytest,wtrain,wcv,wtest
  else:
    return Xtrain,Xcv,Xtest,ytrain,ycv,ytest
def rf_testprep(testfile='test.csv',header=True):
  Xtest = np.zeros((550000,30))
  ids = []
  with open(testfile) as rf:
    if header:
      rf.readline()
    for i in range(550000):
      l = rf.readline()
      if l == '':
        break
      lineid,jets,data,w,o = line_parse(l,'test')
      Xtest[i,:] =  data
      ids.append(lineid)
  return ids, Xtest
def rf_init(n=100):
  rf = RFC(n_estimators = n)
  return rf
def rf_train_and_test(predfile):
  Xtr,Xcv,Xte,ytr,ycv,yte = rf_trainprep()
  rf = rf_init(50)
  rf.fit(Xtr,ytr)
  print 'CV Score: ' + str(rf.score(Xcv,ycv))
  print 'Test Score: ' + str(rf.score(Xte,yte))
  ids,Xtest = rf_testprep()
  rfpreds = rf.predict_proba(Xtest)
  confidence = []
  predictions = []
  with open(predfile,'w') as wf:
    for i in xrange(550000):
      c = rfpreds[i][1]
      if c > .5:
        p = 's'
      else:
        p = 'b'
      wf.write(ids[i]+','+str(c)+','+p+'\n')
  order_predictions(predfile, predfile)
  return

''' ----------------------- '''

''' AdaBoost Functions '''
# use rf_trainprep and rf_testprep for set preparations
def ada_init(n=100):
  ada = ABC(n_estimators = n)
  return ada
def ada_train_and_test(n,predfile):
  # Trains adaboost model on the whole training set with n trees
  # Predicts test set with the model, then formats the file for submission
  Xtr,Xcv,Xte,ytr,ycv,yte = rf_trainprep()
  ada = ada_init(n)
  ada.fit(Xtr,ytr)
  ids,Xtest = rf_testprep()
  adapreds = ada.predict_proba(Xtest)
  confidence = []
  predictions = []
  with open(predfile,'w') as wf:
    for i in xrange(550000):
      c = adapreds[i][1]
      if c > .5:
        p = 's'
      else:
        p = 'b'
      wf.write(ids[i]+','+str(c)+','+p+'\n')
  order_predictions(predfile, predfile)
  return

''' ------------------ '''


def J_SVM(X,y): # FINISH AND TEST
  clf = svm.SVC(C=10,kernel='linear')
  clf.fit(X,y)
  return

# J_smoother compresses the array of costs J by averaging size f clumps
def J_smoother(J_array, f):
  new_array = []
  while len(J_array) > f:
    new_array.append(float(sum(J_array[0:f])) / f)
    J_array = J_array[f:]
  return new_array


def simple_accuracy_check(predfile):
  # prints accuracy measures for an outcome-labeled prediction file
  tfpna = [0,0,0,0,0]
  with open(predfile) as tf:
    while True:
      l = tf.readline()
      if l == '':
        break
      if 's,s' in l:
        tfpna[0] += 1
      elif 's,b' in l:
        tfpna[1] += 1
      elif 'b,s' in l:
        tfpna[2] += 1
      else:
        tfpna[3] += 1
    tfpna[4] = tfpna[0]+tfpna[3]
  c = float(sum(tfpna[:4]))
  for i in range(5):
    tfpna[i] /= c
  precision = float(tfpna[0]) / (tfpna[0]+tfpna[1])
  recall = float(tfpna[0]) / (tfpna[0]+tfpna[2])
  F1 = 2*precision*recall / (precision + recall)
  print 'Accuracy of Prediction file: '+predfile
  print 'True positive %:    ' + str(tfpna[0])
  print 'False positive %:   ' + str(tfpna[1])
  print 'False negative %:   ' + str(tfpna[2])
  print 'True negative %:    ' + str(tfpna[3])
  print 'Accuracy %:         ' + str(tfpna[4])
  print 'Precision:          ' + str(precision)
  print 'Recall:             ' + str(recall)
  print 'F1 score:           ' + str(F1)
  return

# test_accuracy returns measures of accuracy in an outcome-labeled prediction
# file. If precrec == True, it returns precision, recall, and F1.
# If precrec == False, it returns tp,fp,fn,tn,acc.
def test_accuracy(predfile, precrec=True):
  tfpna = [0,0,0,0,0]
  with open(predfile) as tf:
    while True:
      l = tf.readline()
      if l == '':
        break
      if 's,s' in l:
        tfpna[0] += 1
      elif 's,b' in l:
        tfpna[1] += 1
      elif 'b,s' in l:
        tfpna[2] += 1
      else:
        tfpna[3] += 1
    tfpna[4] = tfpna[0]+tfpna[3]
  c = float(sum(tfpna[:4]))
  if precrec:
    precision = float(tfpna[0]) / (tfpna[0]+tfpna[1])
    recall = float(tfpna[0]) / (tfpna[0]+tfpna[2])
    F1 = 2*precision*recall / (precision + recall)
    tfpna = [precision, recall, F1]
  else:
    for i in range(5):
      tfpna[i] /= c
  return tfpna

# model_logit_8bins trains a simple logistic regression model on each of
# the 8 jet_bins
def model_logit_8bins():
  norms = {}
  thetas = {}
  for k in jet_bins:
    # initialize model
    fname = 'train_'+k+'.csv'
    print 'Getting file size: ' + fname
    m = get_file_length(fname)
    print 'Loading data from ' + fname
    X,w,y = load_jetsplit_data(fname,m)
    # get normalizing factors, normalize data
    print 'Normalizing data from ' + fname
    mu,sigma = get_mu_sigma(X)
    normalize_all(X,mu,sigma)
    mu = row_mat_to_list(mu)
    sigma = row_mat_to_list(sigma)
    norms[k] = {'mu':mu, 'sigma':sigma}
    # train each bin's training set with a theta-optimized logit
    print 'Training model for ' + fname
    theta = optimized_theta_STL(X,y,50)
    # store final theta in thetas.json
    theta = row_mat_to_list(np.transpose(theta))
    thetas[k] = theta
  print 'Dumping norms and thetas'
  with open('norms.json','w') as nf:
    json.dump(norms,nf)
  with open('thetas.json','w') as tf:
    json.dump(thetas,tf)
  return


if __name__ == '__main__':
  print "bake 'em away, toys."

  ada_train_and_test(500,'ada_pred_500t.csv')

  #how_many_trees()

  #rf_train_and_test('rf_pred.csv')

  '''hrecord = test_predict_NN_8bins('training.csv','nn_8_train_pred.csv',True,True)
  simple_accuracy_check('nn_8_train_pred.csv')
  simple_histo(hrecord)

  hrecord = test_predict_NN_8bins('test.csv','nn_8_test_pred.csv',False,True)
  simple_histo(hrecord)'''

  '''' training NNs for each jet bin, alternating over time and saving parameters
  while True:
    for k in jet_bins:
      X,w,y = initialize_model('Data/all_'+k+'.csv',jet_bin_samples[k])
      nn = NN(1,60,'jet'+k)
      nn.training_setup(X,y)
      nn.train(1000,.3,100,True)
      nn.saveThetas()
  #j_plot_with_cv(nn.Jtrain,nn.Jcv,True,'firstNNtrain')'''


  #histo_features('train_0.csv',False,feature_array=[1,2,3])

  '''model_logit_8bins()
  predict_all_logit('test.csv','jet_8_pred1.csv',False,.5,True)
  order_predictions('jet_8_pred1.csv','jet_8_pred1_ordered.csv')'''

  '''tp=[]
  fp=[]
  fn=[]
  tn=[]
  acc=[]
  for k in jet_bins:
    p = 'Data/cv_'+k+'_pred.csv'
    tfpna = test_accuracy(p,False)
    tp.append(tfpna[0])
    fp.append(tfpna[1])
    fn.append(tfpna[2])
    tn.append(tfpna[3])
    acc.append(tfpna[4])
  plot_accuracies(tp,fp,fn,tn,acc)'''


  '''y = []
  prev = [] # load from file if using this often
  while True:
    ab = two_feature_random_plotter('training.csv',[],n500,True)
    prev.append(ab)
    yn = raw_input('Worth investigating? y/n/yq/nq: ')
    if yn == 'y':
      y.append(ab)
    if yn[-1] == 'q':
      break
  print y'''

  '''tp = []
  fp = []
  fn = []
  tn = []
  trange = range(30,72)
  for threshold in trange:
    t = threshold/100.
    predict_all_logit('jet_0_test.csv','jet_0_pred.csv',True,t)
    acc = test_accuracy('jet_0_pred.csv')
    tp.append(acc[0])
    fp.append(acc[1])
    fn.append(acc[2])
    tn.append(acc[3])
  plot_threshold(trange,tp,fp,fn,tn)'''
