import cPickle, gzip
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

from PassiveAggressiveMaxOut import *
from utils import zscore_data

ds = "MNIST.pkl.gz"

print "Dataset: " + ds

# Load the dataset
f = gzip.open('../DATASETS/' + ds, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train, Y_train = train_set
X_valid, Y_valid = valid_set
X_test, Y_test = test_set

class_1 = 3
class_2 = 8

train_indexes = np.any([Y_train == class_1, Y_train == class_2], axis=0)
test_indexes = np.any([Y_test == class_1, Y_test == class_2], axis=0)

X = X_train[train_indexes, :]
Y = Y_train[train_indexes]

X_t = X_test[test_indexes, :]
Y_t = Y_test[test_indexes]

Y[np.where(Y == class_1)] = -1
Y[np.where(Y == class_2)] = 1

Y_t[np.where(Y_t == class_1)] = -1
Y_t[np.where(Y_t == class_2)] = 1

tr_number_of_class_1 = np.sum((Y == -1))
tr_number_of_class_2 = np.sum((Y == 1))

te_number_of_class_1 = np.sum((Y_t == -1))
te_number_of_class_2 = np.sum((Y_t == 1))

print "# of " + str(class_1) + " (tr): " + str(tr_number_of_class_1)
print "# of " + str(class_2) + " (tr): " + str(tr_number_of_class_2)
print "Total (tr): " + str(tr_number_of_class_1 + tr_number_of_class_2)
print "# of " + str(class_1) + " (te): " + str(te_number_of_class_1)
print "# of " + str(class_2) + " (te): " + str(te_number_of_class_2)
print "Total (te): " + str(te_number_of_class_1 + te_number_of_class_2)

n_samples_tr = X.shape[0]
n_samples_te = X_t.shape[0]

print "Training: " + str(n_samples_tr)
print "Test: " + str(n_samples_te)
print "======Range======="
print "Max. value = " + str(np.max(np.max(X_train)))
print "Min. value = " + str(np.min(np.min(X_train)))
print "Max. value = " + str(np.max(np.max(X_test)))
print "Min. value = " + str(np.min(np.min(X_test)))

zscore = True

if zscore:
    X, mean, std = zscore_data(X)

    X_t, _, _ = zscore_data(X_t, mean, std)
    print "======Range======="
    print "(After zscore)"
    print "Max. value = " + str(np.max(np.max(X)))
    print "Min. value = " + str(np.min(np.min(X)))
    print "Max. value = " + str(np.max(np.max(X_t)))
    print "Min. value = " + str(np.min(np.min(X_t)))

rep = 20

always_update = False

err_list = []

err_rate_list = []

for r in xrange(rep):
    print "---(rep #" + str(r) + ")---"

    # Permute
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    Y = Y[perm]

    c_0 = 0.1
    c_1 = c_0
    alpha = 0.9
    hidden_units = 16
    pieces = 2
    a = -0.1
    b = 0.1

    clf = PassiveAggressiveMaxOut(c_0=c_0, c_1=c_1, alpha=alpha, pieces=pieces, hidden_units=hidden_units, always_update=always_update, a=a, b=b, debug=True)

    n_samples = n_samples_tr

    # For initiating weights and things... (workaround)
    clf.init_weights(input_units=X.shape[1], pieces=clf.pieces, hidden_units=clf.hidden_units)

    clf.inverse_label_mapping = {-1: -1, 1: 1}
    w0 = clf.w_0
    w1 = clf.w_1

    start = time.time()

    counter_tr = 0

    start = time.time()

    test_error_rate_by_samples = []

    for n_sample in xrange(n_samples):

        x_t = X[n_sample, :]
        y_t = Y[n_sample]

        y_p = clf.fit_one_sample(x_t, y_t, w0, w1)

        if y_p != y_t:
            counter_tr += 1

        if n_sample % 500 == 0 or n_sample == n_samples - 1:
            Y_pred = clf.predict_batch(X_t)

            errs = np.sum(Y_pred != Y_t)

            test_error_rate_by_samples.append((100.0 * errs) / n_samples_te)

    err_list.append(errs)
    err_rate_list.append(test_error_rate_by_samples[-1])

    print("Cumulative mistake rate: " + str((100.0 * counter_tr) / n_samples)) + " %"
    print("--- %s seconds (tr)---" % ((time.time() - start)))

print "Mean: " + str(np.mean(err_rate_list)) + " +/- " + str(np.std(err_rate_list)) + ", hidden_units= " + str(
    hidden_units) + ", pieces= " + str(pieces) + ", alpha= " + str(alpha) + ", C= " + str(c_0)

print "Mean: " + str(100.0 - np.mean(err_rate_list)) + " +/- " + str(np.std(err_rate_list))
