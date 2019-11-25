import numpy as np
import astropy.units as u
try:
    from astropy.timeseries import BoxLeastSquares
except:
    print('WARNING: astropy.timeseries unavailable, upgrade astropy (requires python 3.6+)')

try:
    import cuvarbase.bls as bls
except:
    print('WARNING: cuvarbase not found, fast BLS not available')



def run_gatspy_multiband(lc,pmin=0.01,pmax=100.):
    model = periodic.LombScargleMultibandFast(fit_period=True)
    model.optimizer.period_range=(pmin, pmax)
    model.fit(lc[:,0], lc[:,1], lc[:,2], lc[:,3])

    return model.best_period



def BLS_preproc(data,reject_outliers=False):
    """ run preprocessing before running bls. This function combined the
    multiband lightcurves by subtracting the median and remove high outliers 
    per band
    """

    # preproc data before fitting; convert to flux and normalise:
    t = data[:,0]
    y = data[:,1]
    dy = data[:,2]
    f = data[:,3]

    # normalise
    for k,fid in enumerate(np.sort(np.unique(f))):
        m = f==fid        
        med = np.nanmedian(y[m])
        y[m]/=med
        dy[m]/=med

    if reject_outliers:
        o = np.zeros_like(t)
        # reject outliers
        for k,fid in enumerate(np.sort(np.unique(f))):
            m = f==fid        
            d40p = np.percentile(y[m],90)-np.median(y[m])
            #print(d40p)
            o[m] += y[m] > (np.median(y[m])+5*d40p)
            
        
        o = (o>0)
        print("removed %d outliers" %np.sum(o))
        t = t[~o]
        y = y[~o]
        dy = dy[~o]
        f = f[~o]

    return np.c_[t,y,dy,f]



def BLS_solution(lc,p,qmin=0.01,qmax=0.1,dlogq=0.1):
    t = lc[:,0]
    y = lc[:,1]
    dy = lc[:,2]

    # set up search parameters
    search_params = dict(qmin=qmin,
                     qmax=qmax,
                     # The logarithmic spacing of q
                     dlogq=dlogq,
                     # Number of overlapping phase bins
                     # to use for finding the best phi0
                     noverlap=3)
    bls_power,sols = bls.eebls_gpu(t, y, dy, [p**-1,],
                                **search_params)
    return bls_power[0],sols[0][0],sols[0][1]


def run_BLScuvarbase_search(lc,pmin=30./60/24.,pmax=3.,oversampling=3.,
    qmin=0.01,qmax=0.1,dlogq=0.1):
    """Run the cuvarbase BLS method.
    """

    t = lc[:,0]
    t_min = np.min(t)
    t = t - t_min
    y = lc[:,1]
    dy = lc[:,2]

    # set up search parameters
    search_params = dict(qmin=qmin,
                     qmax=qmax,
                     # The logarithmic spacing of q
                     dlogq=dlogq,
                     # Number of overlapping phase bins
                     # to use for finding the best phi0
                     noverlap=3)

    baseline = max(t) - min(t)
    df = search_params['qmin'] / baseline / oversampling
    fmin = (pmax)**-1
    fmax = (pmin)**-1

    nf = int(np.ceil((fmax - fmin) / df))
    #print('searching %d frequencies' %nf)
    freqs = fmin + df * np.arange(nf)

    bls_power = bls.eebls_gpu_fast(t, y, dy, freqs,
                                **search_params)



    return freqs**-1,bls_power



def run_BLScuvarbase(lc,pmin=30./60/24.,pmax=3.,oversampling=3.,
        qmin=0.01,qmax=0.1,dlogq=0.1,reject_outliers=False):
    """ Run BLS periodsearch using cuvarbase. 

    input:
    lc : array 
    """

    # preproc
    lc = BLS_preproc(lc,reject_outliers)



    #
    periods, power = run_BLScuvarbase_search(lc,pmin,pmax,oversampling,
                        qmin,qmax,dlogq)

    # need a better check for this...
    p = np.argmax(bls_power)

    # not optimal, copies data to GPU a second time...
    p,t0,q = BLS_solution(lc,p,qmin,qmax,dlogq)

    return p,t0+t_min,q,periods,power



