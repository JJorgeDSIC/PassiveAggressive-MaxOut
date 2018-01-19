import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import datetime

def zscore_data(X, mean=None, st=None, totsd=None):

    num_samples = X.shape[0]
    dim = X.shape[1]
    data = X

    if mean is None and st is None and totsd is None:

        mean = np.mean(data, axis=0)

        st = np.std(data, axis=0)

        totmean = np.sum(np.sum(data))

        totmean /= (dim * num_samples)

        totmean_zero = data - totmean

        totmean_zero *= totmean_zero

        totsd = np.sum(np.sum(totmean_zero))

        totsd /= (dim * num_samples)

        totsd = np.sqrt(totsd)

        mean_zero = data - mean

        for i in xrange(dim):

            if st[i] != 0:
                data[:, i] = mean_zero[:, i] * 1.0 / st[i]
            else:
                data[:, i] = mean_zero[:, i] * 1.0 / totsd

    else:

        totmean = np.sum(np.sum(data))

        totmean /= (dim * num_samples)

        totmean_zero = data - totmean

        totmean_zero *= totmean_zero

        totsd = np.sum(np.sum(totmean_zero))

        totsd /= (dim * num_samples)

        totsd = np.sqrt(totsd)

        mean_zero = data - mean

        for i in xrange(dim):

            if st[i] != 0:
                data[:, i] = mean_zero[:, i] * 1.0 / st[i]
            else:
                data[:, i] = mean_zero[:, i] * 1.0 / totsd

    return data, mean, st


####################DRAWING METHODS############################
def plot_points(X, Y, name='plot'):
    f = plt.figure(1)

    f.set_size_inches(15, 10)

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.axes().set_aspect('equal')

    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', s=500, cmap=plt.cm.Paired, c='blue')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='^', s=500, cmap=plt.cm.Paired, c='red')

    #plt.savefig(name + str(datetime.datetime.now().isoformat()) + '.png', bbox_inches='tight')
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.clf()

def plot_frontiers(X, Y, clf, n_samples=-1, filename='test', fixed=None):
    h = .2  # step size in the mesh

    # create a mesh to plot in
    if fixed is not None:

        x_min, x_max = fixed[0], fixed[1]
        y_min, y_max = fixed[2], fixed[3]

    else:

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min - 0.5, x_max + 0.5, h),
                         np.arange(y_min - 0.5, y_max + 0.5, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = np.zeros((points.shape[0], 1))

    for p in xrange(points.shape[0]):
        Z[p], _ = clf.predict_one_sample(points[p, :])

    Z = Z.reshape(xx.shape)

    f = plt.figure(1)

    f.set_size_inches(15, 10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.axes().set_aspect('equal')

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', s=500, cmap=plt.cm.Paired, c='blue')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='^', s=500, cmap=plt.cm.Paired, c='red')

    f.savefig(filename + '.png', bbox_inches='tight')

    f.clf()

    del f

    plt.close()


def plot_both_spaces(X, Y, w_0_prev, w_1_prev, w_0, w_1, clf, filename='test', bidim=True, fixed=None):
    
    X_inner = X
    Y_inner = Y

    n_samples = X.shape[0]

    print n_samples        
    
    f_inner, axarr_inner = plt.subplots(1, 2)
    
    axarr_inner[0].set_aspect('equal', 'datalim')
    axarr_inner[1].set_aspect('equal', 'datalim')

    f_inner.set_size_inches(30, 15)

    axarr_inner[0].set_aspect(1)
    axarr_inner[1].set_aspect(1)
    
    if fixed is not None:
        
        x_min, x_max = fixed[0], fixed[1]
        y_min, y_max = fixed[2], fixed[3]
        
    else:

       
        x_min, x_max = X_inner[:, 0].min() - 1, X_inner[:, 0].max() + 1
        y_min, y_max = X_inner[:, 1].min() - 1, X_inner[:, 1].max() + 1
    
    if bidim:
      
        # First plot, original space
        h = .2  # step size in the mesh
        
        xx, yy = np.meshgrid(np.arange(x_min-1, x_max+1, h),
                             np.arange(y_min-1, y_max+1, h))
        
        
        points=np.c_[xx.ravel(), yy.ravel()]
        
        Z = np.zeros((points.shape[0],1))
        
        
        for p in xrange(points.shape[0]):
            Z[p], _ = clf.predict_one_sample(points[p, :], w_0, w_1)
            
        Z=Z.reshape(xx.shape) 
        

        axarr_inner[0].set_xlim(x_min, x_max)
        axarr_inner[0].set_ylim(y_min, y_max)
        
        axarr_inner[0].contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2, c=Y, norm=clrs.Normalize(vmin=-1.0, vmax=1.0))

        axarr_inner[0].scatter(X_inner[Y_inner == -1, 0], X_inner[Y_inner == -1, 1], marker='o', s=500, cmap=plt.cm.Paired, c='blue')
        axarr_inner[0].scatter(X_inner[Y_inner == 1, 0], X_inner[Y_inner == 1, 1], marker='^', s=500, cmap=plt.cm.Paired, c='red')

        if Y_inner[-1] == -1:
            axarr_inner[0].scatter(X_inner[-1, 0], X_inner[-1, 1], marker='o', s=700, cmap=plt.cm.Paired, c='green')
        else:
            axarr_inner[0].scatter(X_inner[-1, 0], X_inner[-1, 1], marker='^', s=700, cmap=plt.cm.Paired, c='green')


   
    # Second plot, projected space
    
    samples_projected_prev = np.zeros((n_samples, 2), dtype=np.float32)
    
    for n_s in xrange(n_samples):
        # print n_s
        samples_projected_prev[n_s,:],_ = clf.project_one_sample(X[n_s,:], w_0_prev, w_1_prev)
        # _, samples_projected_prev[n_s,:]= clf.project_one_sample(X[n_s,:], w_0_prev, w_1_prev)

        
    samples_projected = np.zeros((n_samples, 2), dtype=np.float32)
    
    for n_s in xrange(n_samples):
    
        samples_projected[n_s,:],_ =  clf.project_one_sample(X[n_s,:], w_0, w_1)
        # _, samples_projected[n_s,:] =  clf.project_one_sample(X[n_s,:], w_0, w_1)


    print "===="
        
    x_min, x_max = samples_projected[:, 0].min() - 1, samples_projected[:, 0].max() + 1
    y_min, y_max = samples_projected[:, 1].min() - 1, samples_projected[:, 1].max() + 1   
    
    axarr_inner[1].set_xlim(-2.0, 2.0)
    axarr_inner[1].set_ylim(-2.0, 2.0)
    
    axarr_inner[1].scatter(samples_projected_prev[Y_inner == -1, 0], samples_projected_prev[Y_inner == -1, 1], marker='o', alpha=0.2, s=200, cmap=plt.cm.Paired, c='blue')
    axarr_inner[1].scatter(samples_projected_prev[Y_inner == 1, 0], samples_projected_prev[Y_inner == 1, 1], marker='^', alpha=0.2, s=200, cmap=plt.cm.Paired, c='red')

    axarr_inner[1].scatter(samples_projected[Y_inner == -1, 0], samples_projected[Y_inner == -1, 1], marker='o', s=200, cmap=plt.cm.Paired, c='blue')
    axarr_inner[1].scatter(samples_projected[Y_inner == 1, 0], samples_projected[Y_inner == 1, 1], marker='^', s=200, cmap=plt.cm.Paired, c='red')

    if Y_inner[-1] == -1:
        axarr_inner[1].scatter(samples_projected_prev[-1, 0], samples_projected_prev[-1, 1], marker='o', alpha=0.2, s=700, cmap=plt.cm.Paired, c='green')
    else:
        axarr_inner[1].scatter(samples_projected_prev[-1, 0], samples_projected_prev[-1, 1], marker='^', alpha=0.2, s=700, cmap=plt.cm.Paired, c='green')


    if Y_inner[-1] == -1:
        axarr_inner[1].scatter(samples_projected[-1, 0], samples_projected[-1, 1], marker='o', s=700, cmap=plt.cm.Paired, c='green')
    else:
        axarr_inner[1].scatter(samples_projected[-1, 0], samples_projected[-1, 1], marker='^', s=700, cmap=plt.cm.Paired, c='green')

    area=20
    #weights
    normalized_w_1_prev = w_1_prev / np.linalg.norm(w_1_prev)

    y_tics = np.arange(-area, area, 0.1)
    
    axarr_inner[1].plot(y_tics, -(y_tics * normalized_w_1_prev[0]) / normalized_w_1_prev[1],'r-')

    normalized_w_1 = w_1 / np.linalg.norm(w_1)

    axarr_inner[1].plot(y_tics, -(y_tics * normalized_w_1[0]) / normalized_w_1[1],'k-')

    f_inner.savefig(filename + '.png', bbox_inches='tight')

    f_inner.clf()

    del f_inner

    plt.close()
    