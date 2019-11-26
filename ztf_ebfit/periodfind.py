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



def run_curvarbase_solution(lc,p,qmin=0.01,qmax=0.1,dlogq=0.1,noverlap=3):

    # prep data
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
                     noverlap=noverlap)
    bls_power,sols = bls.eebls_gpu(t, y, dy, [p**-1,],
                                **search_params)

    power,q,phi0 = bls_power[0],sols[0][0],sols[0][1]

    # calculate t0
    t0 = (np.median(t)//p+(phi0+0.5*q))*p + t_min

    return power,q,t0



def run_BLScuvarbase_search(lc,pmin=30./60/24.,pmax=3.,oversampling=3.,
    qmin=0.01,qmax=0.1,dlogq=0.1,noverlap=3):
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
                     noverlap=noverlap)

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
        qmin=0.01,qmax=0.1,dlogq=0.1,pos_sigmaclip=None):
    """ Run BLS periodsearch using cuvarbase. 

    input:
    lc : array 
    """

    # preproc # THIS NEEDS REVIEWING!!!
    if pos_sigmaclip is not None:
        o = np.zeros_like(t)
        # reject outliers
        for k,fid in enumerate(np.sort(np.unique(f))):
            m = f==fid        
            d40p = np.percentile(y[m],90)-np.median(y[m])
            #print(d40p)
            o[m] += y[m] > (np.median(y[m])+5*d40p)

    #
    period, power = run_BLScuvarbase_search(lc,pmin,pmax,oversampling,
                        qmin,qmax,dlogq)

    # need a better check for this, maybe also get some kind of significance?
    p = period[np.argmax(power)]
    print(p)

    # not optimal, copies data to GPU a second time...
    _power,t0,q = run_curvarbase_solution(lc,p,qmin,qmax,dlogq)
    print(_power,t0,q)

    # calculate significance
    idx = np.argmax(power)
    d = 5001
    i_min = np.max([0,idx-d])
    i_max = np.min([idx+d,np.size(power)])
    sig = power[idx]/np.median(power[i_min:i_max]) 

    return p,t0,q,period,power,sig



