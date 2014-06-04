'''
Pre-processing functions for the Higgs classification dataset.

Anything that happens before modeling and prediction. Mostly file operations.

Dan Morris
5/20/2014 ->
'''
import numpy as np
import simplejson as json
from base import *
import csv, operator

alltrainfiles = ['jet_0_cv.csv','jet_0_train.csv','jet_0_test.csv',\
                 'jet_1_cv.csv','jet_1_train.csv','jet_1_test.csv',\
                 'jet_2_cv.csv','jet_2_train.csv','jet_2_test.csv',\
                 'jet_3_cv.csv','jet_3_train.csv','jet_3_test.csv',\
                 'train_jet0.csv','train_jet1.csv','train_jet2.csv',\
                 'train_jet3.csv','training.csv']

# split_raw_data separates a datafile into two
# completewrite includes all examples without a '-999.0' element
# incompletewrite includes all examples with a '-999.0' element
def split_raw_data(readfile, completewrite, incompletewrite):
  icount = 0 # number of incomplete examples
  ccount = 0 # number of complete examples
  with open(readfile) as rf:
    with open(completewrite, 'w') as wcf:
      with open(incompletewrite, 'w') as wif:
        header = rf.readline()
        #wcf.write(header) - no headers in derived data files!
        #wif.write(header) - no headers in derived data files!
        while True:
          l = rf.readline()
          if l == '':
            break
          if '-999.0' in l:
            wif.write(l)
            icount += 1
          else:
            wcf.write(l)
            ccount += 1
  print 'Complete examples:   ' + str(ccount)
  print 'Incomplete examples: ' + str(icount)
  return

# split_train_8bins splits training set into 8 bins based on jets+feature1
# and then creates train/cv/test sets in each.
# files are stored as 'train_','cv_','test_' + '0','0u','1','1u'... + '.csv'
# or 'all_'+jettype+'.csv'
def split_train_8bins(trainfile='training.csv',splitcv=False):
  binsdict = {}
  for k in jet_bins:
    binsdict[k] = []
  with open(trainfile) as rf:
    rf.readline() # clear header
    for i in xrange(250000):
      l = rf.readline()
      lineid, jets, data, weight, outcome = jet_line_parse(l)
      binsdict[jets].append(l)
  if splitcv:
    for k in jet_bins:
      ntr, ncv, nte = n_sorter(len(binsdict[k]))
      with open('train_'+k+'.csv','w') as trainf:
        for e in xrange(ntr):
          trainf.write(binsdict[k][e])
      binsdict[k] = binsdict[k][ntr:]
      with open('cv_'+k+'.csv','w') as cvf:
        for e in xrange(ncv):
          cvf.write(binsdict[k][e])
      binsdict[k] = binsdict[k][ncv:]
      with open('test_'+k+'.csv','w') as testf:
        for e in xrange(nte):
          testf.write(binsdict[k][e])
  else:
    for k in jet_bins:
      n = len(binsdict[k])
      with open('Data/all_'+k+'.csv','w') as wf:
        for ex in xrange(n):
          wf.write(binsdict[k][ex])
      print 'Jet bin '+k+' count: '+str(n)
  return

# small_csv generates a small csv (first n lines of readfile) for code-testing
def small_csv(readfile, writefile, n=100, killheader=False):
  with open(readfile) as rf:
    if killheader:
      burn = rf.readline()
    with open(writefile, 'w') as wf:
      for i in range(n):
        wf.write(rf.readline())
  return

# example_sorter splits a training set roughly 60/20/20 between training,
# cross-validation, and testing sets
def example_sorter(readfile, trainname, cvname, testname, header=False):
  traincount = 0
  cvcount = 0
  testcount = 0
  with open(readfile) as rf:
    if header:
      rf.readline()
    with open(trainname, 'w') as trf:
      with open(cvname, 'w') as cvf:
        with open(testname, 'w') as tef:
          while True:
            l = rf.readline()
            if l == '':
              break
            r = random()
            if r < .6: #train
              trf.write(l)
              traincount += 1
            elif r < .8: #cv
              cvf.write(l)
              cvcount += 1
            else: #test
              tef.write(l)
              testcount += 1
  print 'Training examples:         ' + str(traincount)
  print 'Cross-Validation examples: ' + str(cvcount)
  print 'Testing examples:          ' + str(testcount)
  return

# n_sorter determines how many are training, cv, test based on count n
def n_sorter(n):
  ntrain = n * 6/10
  n = n - ntrain
  ncv = n / 2
  ntest = n - ncv
  return (ntrain, ncv, ntest)

# shuffle_rows shuffles the example rows of a file (for stochastic GD, perhaps)
def shuffle_rows(readfile, writefile, header=True):
  rows = []
  with open(readfile) as rf:
    if header:
      h = rf.readline()
    while True:
      l = rf.readline()
      if l == '':
        break
      rows.append(l)
  shuffle(rows)
  with open(writefile, 'w') as wf:
    if header:
      wf.write(h)
    for r in rows:
      wf.write(r)
  return

# inspect_element_1 determines the percentage signal of all examples where
# element 1 is undefined
def inspect_element_1(readfile='training.csv', header=True):
  poscases = 0
  signal = 0.
  with open(readfile) as rf:
    if header:
      rf.readline()
    while True:
      l = rf.readline()
      if l=='':
        break
      l = l.split(',')
      if l[1] == '-999.0':
        poscases += 1
        if 's' in l[32]:
          signal += 1
  print str(poscases) + ' positive examples.'
  print str(signal) + ' are classified signal'
  ratio = signal / poscases
  print str(ratio) + ' is the percentage of signal'
  return

# jet_split splits the training set based on number of jets (feature 23)
def jet_split(readfile='training.csv', header=True):
  jk = ['0','1','2','3','_other']
  with open(readfile) as rf:
    if header:
      rf.readline()
    wf = {} # dict of write files, keys are number of jets
    c = {}  # dict of example counts, same keys
    for j in jk:
      wf[j] = open('train_jet'+j+'.csv', 'w')
      c[j] = 0
    while True:
      l = rf.readline()
      if l == '':
        break
      s = l.split(',')
      if s[23] in wf.keys():
        wf[s[23]].write(l)
        c[s[23]] += 1
      else:
        wf['_other'].write(l)
    for j in jk:
      wf[j].close()
  for j in jk:
    print 'Jet count ' + str(j) + ': ' + str(c[j])
  return

# jet_prep runs example_sorter on each jet training file
def jet_prep():
  for j in ['0','1','2','3']:
    trn = 'jet_'+j+'_train.csv'
    cvn = 'jet_'+j+'_cv.csv'
    ten = 'jet_'+j+'_test.csv'
    example_sorter('train_jet'+j+'.csv',trn,cvn,ten)
  return

# weight_info returns information about the weight distribution of a sample
def weight_info(readfile, header=False):
  ws = 0.
  wb = 0.
  cs = 0
  cb = 0
  with open(readfile) as rf:
    if header:
      rf.readline()
    while True:
      l = rf.readline()
      if l == '':
        break
      l = l.split(',')
      w = float(l[31])
      if 's' in l[32]:
        ws += w
        cs += 1
      else:
        wb += w
        cb += 1
  print 'For file ' + readfile + ', the weights are as follows:'
  print 'Total Signal weight:     ' + str(ws)
  print 'Total Background weight: ' + str(wb)
  print 'Count Signal:            ' + str(cs)
  print 'Count Background:        ' + str(cb)
  d = {'ws':ws, 'wb':wb, 'cs':cs, 'cb':cb}
  with open('weights.json') as rf:
    bigdict = json.loads(rf.read())
  bigdict[readfile] = d
  with open('weights.json','w') as wf:
    json.dump(bigdict, wf)
  return

# get_file_length returns the number of lines/rows in file fname
def get_file_length(fname):
  with open(fname) as f:
    c = 0
    while True:
      if f.readline() != '':
        c += 1
      else:
        break
  return c

# order_predictions orders the list of test predictions and formats it properly
# for submission
def order_predictions(predin,predout):
  with open(predin) as f:
    reader = csv.reader(f)
    sl = sorted(reader, key=operator.itemgetter(1), reverse=True)
  # kick tiny exponent numbers from front to back of list
  while 'e' in sl[0][1]:
    kick = sl[0]
    sl = sl[1:] + [kick]
  with open(predout,'w') as f:
    f.write('EventId,RankOrder,Class\n')
    for i in xrange(len(sl)):
      f.write(sl[i][0]+','+str(i+1)+','+sl[i][2]+'\n')
  return

if __name__ == '__main__':
  print 'nothing to see here.'

  order_predictions('jet_8_pred.csv','jet_8_pred_ordered.csv')
