import cPickle, gzip
import time
import matplotlib.pyplot as plt
import datetime
import time
import sys

from PassiveAggressiveMaxOut import *
from utils import zscore_data

if len(sys.argv) < 2:

    raise Exception("python example.py dataset {hidden_units pieces C_0 C_1 alpha version={0=PAMO-I, 1=PAMO-II} partition={training=1, test=2, both=3}}")

ds = sys.argv[1]

if len(sys.argv) == 9:

    print "Using input settings..."
    hidden_units = int(sys.argv[2])
    pieces = int(sys.argv[3])
    c_0 = float(sys.argv[4])
    c_1 = float(sys.argv[5])
    alpha = float(sys.argv[6])
    version = int(sys.argv[7])
    partition = int(sys.argv[8])

else:

    print "Using default settings..."
    hidden_units = 256
    pieces = 2
    c_0 = 0.125
    c_1 = c_0
    alpha = 0.9
    version = 0
    partition = 3

if version == 1:
    always_update = True
else:
    always_update = False

zscore = True

rep = 20

print "Dataset: " + ds

f = gzip.open(ds, 'rb')
train_set, test_set = cPickle.load(f)
f.close()

X_train, Y_train = train_set
X_test, Y_test = test_set

if partition == 1:

    X_full = X_train
    Y_full = Y_train

elif partition == 2:

    X_full = X_test
    Y_full = Y_test

else:

    X_full = np.concatenate((X_train, X_test), axis=0)
    Y_full = np.concatenate((Y_train, Y_test), axis=0)

n_samples_tr = X_full.shape[0]

if zscore:
    X_full, mean, std = zscore_data(X_full)

mistake_rate_list = []

mistake_list = []

timing_list_tr = []

for r in xrange(rep):

    print "---(rep #" + str(r) + ")---"

    perm = np.random.permutation(X_full.shape[0])
    X_full = X_full[perm, :]
    Y_full = Y_full[perm]

    clf = PassiveAggressiveMaxOut(c_0=c_0, c_1=c_1, alpha=alpha, pieces=pieces, hidden_units=hidden_units, always_update=always_update)

    n_samples = n_samples_tr

    start = time.time()

    clf.fit_batch(X_full[:n_samples, :], Y_full[:n_samples])

    mistake_rate_list.append(clf.cumulative_error_rate)
    mistake_list.append(clf.cumulative_error)

    timing_list_tr.append((time.time() - start))

    print("--- %s seconds (tr)---" % (timing_list_tr[-1]))
    print "Accum err: %.2f%%" % (clf.cumulative_error_rate)
    print "Accum mistakes: %i / %i" % (clf.cumulative_error, n_samples) 

mean_tr_time = np.mean(timing_list_tr)

print "=====(\#tr=%i, \#dim=%i)====" % (n_samples, X_train.shape[1])
print "Using: " + ds
print "Cum.Err.Rate= %.2f%%+/-%.2f, mistakes: %i / %i" % (np.mean(mistake_rate_list), np.std(mistake_rate_list), np.mean(mistake_list), n_samples_tr)
print "Conf: HU=%i P=%i ALPHA=%.2f C_0=%.4f C_1=%.4f" % (hidden_units, pieces, alpha, c_0, c_1)
print "Imp.Det: A.U=%s" % (always_update)
print "Timing: TR Mean=%.3f s (%.5f per sample) " % (mean_tr_time, mean_tr_time / n_samples_tr)


