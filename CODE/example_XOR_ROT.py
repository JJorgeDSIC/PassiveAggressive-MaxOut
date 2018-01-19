import cPickle, gzip
import time
import datetime
import sys

from PassiveAggressiveMaxOut import *
from utils import zscore_data, plot_points, plot_frontiers

if len(sys.argv) > 1:
    
    print "Using input settings..."
    hidden_units = int(sys.argv[2])
    pieces = int(sys.argv[3])
    c_0 = float(sys.argv[4])
    c_1 = float(sys.argv[5])
    alpha = float(sys.argv[6])
    
else:

    print "Using default settings..."
    hidden_units = 3
    pieces = 2
    c_0 = 0.125
    c_1 = 0.125
    alpha = 0.9


zscore = True
always_update = True

# Load the dataset
ds = "XOR_ROT.pkl.gz"

print "Dataset: " + ds

f = gzip.open('../DATASETS/' + ds, 'rb')
dataset = cPickle.load(f)
f.close()

train_set = dataset[0]

X_train, Y_train = train_set

n_samples_tr = X_train.shape[0]

if zscore:
    X_train, mean, std = zscore_data(X_train)

#########################################
#########################################
#########################################
#########################################
n_samples = X_train.shape[0]

#moving label range
Y_train = Y_train * 2 - 1


plot_points(X_train[:100, :], Y_train[:100], name='../PLOTS/plot_1_100_')
plot_points(X_train[100:200, :], Y_train[100:200], name='../PLOTS/plot_101_200_')
plot_points(X_train[200:300, :], Y_train[200:300], name='../PLOTS/plot_201_300_')



clf = PassiveAggressiveMaxOut(c_0=c_0, c_1=c_1, alpha=alpha, pieces=pieces, hidden_units=hidden_units, always_update=always_update)

# Initiating weights
clf.init_weights(input_units=X_train.shape[1], pieces=clf.pieces, hidden_units=clf.hidden_units)

w0 = clf.w_0
w1 = clf.w_1

start = time.time()

counter = 0

amount = 100

for n_sample in xrange(n_samples):

    x_t = X_train[n_sample, :]
    y_t = Y_train[n_sample]

    y_pred = clf.fit_one_sample(x_t, y_t, w0, w1)

    if y_pred != y_t:
        counter += 1

    if (n_sample + 1) % amount == 0:
        plot_frontiers(X_train[n_sample - (amount-1):n_sample+1, :], Y_train[n_sample - (amount-1):n_sample+1], clf, n_samples=-1,
                       filename='../PLOTS/test_' + str(n_sample+1) + "_", fixed=[-2, 2, -2, 2])

print("Cumulative mistake rate: " + str((100.0 * counter) / n_samples)) + " %"
print("--- %s seconds (tr)---" % ((time.time() - start)))
