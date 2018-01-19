import cPickle, gzip
import time
import sys

from PassiveAggressiveMaxOut import *
from utils import zscore_data

np.random.seed(0)

if len(sys.argv) < 2:

    raise Exception("python example.py dataset {hidden_units pieces C_0 C_1 alpha version={0=PAMO-I, 1=PAMO-II}}")

ds = sys.argv[1]

if len(sys.argv) == 8:

    print "Using input settings..."
    hidden_units = int(sys.argv[2])
    pieces = int(sys.argv[3])
    c_0 = float(sys.argv[4])
    c_1 = float(sys.argv[5])
    alpha = float(sys.argv[6])
    version = int(sys.argv[7])

else:

    print "Using default settings..."
    c_0 = 0.125
    c_1 = c_0
    alpha = 0.9
    hidden_units = 64
    pieces = 2
    version = 0

if version == 1:
    always_update = True
else:
    always_update = False

zscore = True

rep = 20

print "Selected Dataset: " + ds

# Load the dataset
f = gzip.open(ds, 'rb')
train_set, test_set = cPickle.load(f)
f.close()

X_train, Y_train = train_set
X_test, Y_test = test_set

n_samples_tr = X_train.shape[0]
n_samples_te = X_test.shape[0]

if zscore:
    X_train, mean, std = zscore_data(X_train)

    X_test, _, _ = zscore_data(X_test, mean, std)

err_list = []

err_rate_list = []

mistake_rate_list = []

timing_list_tr = []

timing_list_pr = []

for r in xrange(rep):

    print "---(rep #" + str(r) + ")---"

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm, :]
    Y_train = Y_train[perm]

    clf = PassiveAggressiveMaxOut(c_0=c_0, c_1=c_1, alpha=alpha, pieces=pieces, hidden_units=hidden_units, always_update=always_update)
    n_samples = n_samples_tr

    start = time.time()

    clf.fit_batch(X_train[:n_samples, :], Y_train[:n_samples])

    mistake_rate_list.append(clf.cumulative_error_rate)

    timing_list_tr.append((time.time() - start))

    print("--- %s seconds (tr)---" % (timing_list_tr[-1]))

    start = time.time()

    Y_pred = clf.predict_batch(X_test)

    timing_list_pr.append((time.time() - start))

    print("--- %s seconds (pr)---" % (timing_list_pr[-1]))

    errs = np.sum(Y_pred != Y_test)

    err_list.append(errs)
    err_rate_list.append((errs * 100.0) / n_samples_te)

    print "Errs: %i / %i" % (errs, n_samples_te)
    print "Errs rate: %.2f%% || Acc rate: %.2f%%" % ((errs * 100.0) / n_samples_te, 100.0 - ((errs * 100.0) / n_samples_te))
    print "Accum err: %.2f%%" % (clf.cumulative_error_rate)


mean_err_rate = np.mean(err_rate_list)
mean_tr_time = np.mean(timing_list_tr)
mean_pr_time = np.mean(timing_list_pr)

print "=====(\#tr=%i, #te=%i, \#dim=%i)====" % (n_samples_tr, n_samples_te, X_train.shape[1])
print "Using: " + ds
print "Mean: E=%.2f%%+/-%.2f A=%.2f%% Cum.Err.Rate= %.2f%%+/-%.2f" % (mean_err_rate, np.std(err_rate_list), (100.0 - mean_err_rate), np.mean(mistake_rate_list), np.std(mistake_rate_list))
print "Conf: HU=%i P=%i ALPHA=%.2f C_0=%.4f C_1=%.4f" % (hidden_units, pieces, alpha, c_0, c_1)
print "Imp.Det: A.U=%s" % (always_update)
print "Timing: TR Mean=%.3f s (%.5f per sample) PR Mean=%.3f s (%.5f per sample)" % (mean_tr_time, mean_tr_time / n_samples_tr, mean_pr_time, mean_pr_time / n_samples_te)