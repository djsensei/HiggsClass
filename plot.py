import matplotlib.pyplot as plt
import numpy as np

'''TODO : update all save methods to match j_plot'''

def j_plot(j, show=False, savename=''):
  plt.plot(j)
  plt.ylabel('Cost J over iterations')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def j_plot_with_cv(j_train, j_cv, show=False, savename=''):
  i = len(j_train)
  x = np.arange(i)
  plt.plot(x,j_train,'r', label='Training')
  plt.plot(x,j_cv,'b', label='Cross-Validation')
  plt.ylabel('Cost J over iterations')
  plt.legend(loc='upper right')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def acc_plot_with_cv(tp_t,fp_t,fn_t,tn_t,tp_c,fp_c,fn_c,tn_c, show=False, savename=''):
  i = len(tp_t)
  x = np.arange(i)
  plt.plot(x, tp_t, color='#00ff00', label='Train True Pos')
  plt.plot(x, fp_t, color='#ff0000', label='Train False Pos')
  plt.plot(x, fn_t, color='#ff9900', label='Train False Neg')
  plt.plot(x, tn_t, color='#0099cc', label='Train True Neg')
  plt.plot(x, tp_c, color='#00ff99', label='CV True Pos')
  plt.plot(x, fp_c, color='#ff3300', label='CV False Pos')
  plt.plot(x, fn_c, color='#ffff00', label='CV False Neg')
  plt.plot(x, tn_c, color='#0066cc', label='CV True Neg')
  plt.ylabel('Accuracy over iterations')
  plt.legend(loc='upper left')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def plot_learning_curves(mrange, Jtrain, Jcv, show=False, savename=''):
  plt.plot(mrange, Jtrain, 'r', label='Training')
  plt.plot(mrange, Jcv, 'b', label='CV')
  plt.ylabel('Cost J')
  plt.xlabel('Number of Examples m')
  plt.legend(loc='upper right')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def plot_threshold(trange,tp,fp,fn,tn, show=False, savename=''):
  plt.plot(trange,tp,color='#00ff00',label='TP')
  plt.plot(trange,fp,color='#ff0000',label='FP')
  plt.plot(trange,fn,color='#ff9900',label='FN')
  plt.plot(trange,tn,color='#0099cc',label='TN')
  plt.ylabel('TF PN vs. Threshold')
  plt.legend(loc='upper left')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def plot_accuracies(tp,fp,fn,tn,acc, show=False, savename=''):
  l = len(tp)
  r = range(l)
  plt.plot(r,tp,color='#00ff00',label='TP')
  plt.plot(r,fp,color='#ff0000',label='FP')
  plt.plot(r,fn,color='#ff9900',label='FN')
  plt.plot(r,tn,color='#0099cc',label='TN')
  plt.plot(r,acc,color='#000000',label='Acc')
  plt.ylabel('TFPN + Accuracy')
  plt.xlabel('jet_bins')
  plt.legend(loc='upper left')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def plot_precrec(prec, rec, f1, show=False, savename=''):
  l = len(prec)
  r = range(l)
  plt.plot(r,prec,color='#00ff00',label='Precision')
  plt.plot(r,rec,color='#0099cc',label='Recall')
  plt.plot(r,f1,color='#000000',label='F1 Score')
  plt.ylabel('Accuracy scores')
  plt.xlabel('jet_bins')
  plt.legend(loc='lower left')
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

# two_features_plot plots one feature on the x axis and the other on the y
# signal = red, background = black
def two_features_plot(f1b,f2b,f1s,f2s,label1,label2, show=False, savename=''):
  plt.plot(f1b,f2b,color='#000000',marker='.',label='background',linestyle='None')
  plt.plot(f1s,f2s,color='#ff0000',marker='.',label='signal',linestyle='None')
  plt.xlabel(label1)
  plt.ylabel(label2)
  if savename != '':
    plt.savefig('Plots/'+savename+'.png')
  if show:
    plt.show()
  return

def simple_histo(f,bins=20):
  plt.hist(f,bins)
  plt.show()
  return

# binary_histo plots a single feature in a histogram, splitting True and False
# elements into separate sets
def binary_histo(t,f,bins=50):
  plt.hist([t,f],bins,histtype='barstacked')
  plt.show()
  return
