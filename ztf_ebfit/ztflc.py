import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, brute, basinhopping
import copy
import emcee

from .periodfind import run_BLScuvarbase
from .utils import flux2mag, mag2flux
from .ebmodels import EBmodel_multiband
#import .lcmodels as lcmodels

"""import periodfind"""


class Ztflc:
    """ Class to store and represent ZTF lightcurves of a single object. 
    """

    def __init__(self,filename=None,data=None):
        # set some variables
        self.p = 0 # period
        self.t0 = 0 # mideclipse time
        self.dur = 0.1 # eclipse dutycycle
        self.colours = dict({1:'g',2:'r',3:'k',})
        self.EBmodel = {} # storage for EBmodel parameters
        self.EBmodel['filtpars'] = {}

        if filename is not None and data is not None:
            print('Cannot give both filename AND data, choice 1')
        elif filename is not None:
            self.preproc(np.loadtxt(filename))
        elif data is not None:
            self.preproc(data)
        pass 

    def preproc(self,data):
        self.t = data[:,0]
        self.y = data[:,1]
        self.dy = data[:,2]
        self.fid = data[:,3]
        self.pid = data[:,4]
        self.ra = data[:,5]
        self.dec = data[:,6]
        self.alert = data[:,7]
        self.flag = data[:,8]

        # store the coordinates
        self.ra_med = np.nanmedian(self.ra)
        self.dec_med = np.nanmedian(self.dec)
        self.Rmag = flux2mag(np.nanmedian(self.y[self.fid==2]))
        self.Gmag = flux2mag(np.nanmedian(self.y[self.fid==1]))

        # normalise data
        self.normalise_data()

    def normalise_data(self):
        yn = copy.deepcopy(self.y)
        dyn = copy.deepcopy(self.dy)

        self.scaling_factors = dict()
        for n in np.unique(self.fid):
            m = (self.fid==n)
            med = np.median(yn[m])
            self.scaling_factors[n] = med
            yn[m] /= med
            dyn[m] /= med

        self.yn = yn
        self.dyn = dyn



    def run_BLSsearch(self,pmin=0.02,pmax=5.,filters=[1,2],pos_sigmaclip=5,
            clean=True,alerts=True,oversampling=3.,qmin=0.01,qmax=0.1,
            dlogq=0.1,noverlap=3):

        # make datamask
        mask = np.isin(self.fid,filters)
        if clean:
            mask *= (self.flag == 0)
        if not alerts:
            mask *= (self.alert == 0)

        # the lc to search
        lc = np.c_[self.t,self.yn,self.dy][mask]

        # run search
        p,t0,q,period,power,sig = run_BLScuvarbase(lc,pmin=pmin,pmax=pmax,
            oversampling=oversampling,qmin=qmin,qmax=qmax,dlogq=dlogq)

        # store results
        self.p = p
        self.t0 = t0
        self.dur = q
        self.BLS = dict()
        self.BLS['p'] = p
        self.BLS['t0'] = t0
        self.BLS['dur'] = q
        self.BLS['sig'] = sig
        self.BLS['power'] = power
        self.BLS['period'] = period



    def fit_EBmodel(self,filters=[1,2],verbose=False,clean=True,alerts=True,
            mcmc=0):
        # given a period and t0, fit a simple EBmodel 
        # THE NORMALISED DATA WILL BE FITTED!
        # select the best fitting model

        # set initial values
        p,t0 = self.p,self.t0
        t,y,dy,fid = self.t,self.yn,self.dyn,self.fid # FIT NORMALISED DATA

        # make datamask
        mask = np.isin(self.fid,filters)
        if clean:
            mask *= (self.flag == 0)
        if not alerts:
            mask *= (self.alert == 0)

        # the lc to fit
        lc = np.c_[self.t,self.yn,self.dy,self.fid][mask]

        q = self.dur # this should output from a BLS fit
                
        # the base parameters
        try: 
            x0 = self.EBmodel['basepars']
        except:
            x0 = [p,t0,0.7*q,0.3*q]
        bounds = [[(1-10**-5)*p,(1+10**-5)*p,],
                  [t0-0.01*p,t0+0.01*p],[0,0.25],[0.,0.25]]
        x_scale = [0.5*10**-6*p,0.0001,0.1*q,0.1*q]

        # the filter parameters
        for k,b in enumerate(filters):
            if b in self.EBmodel['filtpars'].keys():
                xp = self.EBmodel['filtpars'][b]
            else:
                lcmin = np.min(y[b==fid])        
                xp = [1.,(1.-np.max([lcmin,1.])),0.1,0.01,0.01]
            boundsp = np.array([[0.5,1.5],np.array([0,1.1]),np.array([0,1.1]),
                                [0,1.],[0,1.]])
            # add 
            x0 = np.r_[x0,xp]
            bounds = np.r_[bounds,boundsp]
            x_scale = np.r_[x_scale,np.array([0.1,0.1,0.1,0.01,0.01])]

        # RUN LSTQ FIT
        func = lambda pars: ((y-EBmodel_multiband(pars,t,fid))/dy)
        output = least_squares(func,x0,bounds=bounds.T,x_scale=x_scale,
            verbose=verbose)

        self.EBmodel['basepars'] = output.x[:4]
        for n,k in enumerate(filters):
            self.EBmodel['filtpars'][k] = output.x[4+(n*5):4+((n+1)*5)]

        self.EBmodel

        self.trap = dict()
        self.trap['output'] = output
        #self.trap['output2'] = output2.x
        self.trap['my'] = EBmodel_multiband(output.x,t,fid)



        if nmcmc>0:
            def lnp(pars):
                #print('running')
                return -0.5*np.sum(func(pars)**2)

            nwalkers = 128
            ndim = len(output.x)
            pos = output.x + x_scale * 10**-2 * np.random.randn(nwalkers, ndim)            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp,)
            sampler.run_mcmc(pos, mcmc)    

            self.trap['sampler'] = sampler            



    def plot_data_model(self,pars,filters=None):

        if filters==None:
            filters=[1,2,3]

        #pars = self.trap['output'].x
        p = pars[0]
        t0 = pars[1]

        for f in filters:
            m = (self.fid==f)
            
            plt.figure()
            plt.errorbar((self.t-t0)[m]/p%1,self.yn[m],self.dyn[m],
                marker='.', ls='none',c=self.colours[f])
            _t = np.linspace(t0,t0+p,1000,endpoint=False)
            _model = EBmodel_multiband(pars,_t,f*np.ones_like(_t),filters=filters)
            print(_model)
            plt.plot((_t-t0)/p%1,_model,'k-')
            
        plt.show()



    def refine_ephem_BLS(self,filters=[1,2],logtrange=-5,dur=None):
        import astropy.units as u
        from astropy.timeseries import BoxLeastSquares

        # pmin,pmax
        pmin = self.p*(1-10**logtrange),
        pmax = self.p*(1+10**logtrange),
        
        # preproc
        t_med = np.median(self.t)
        _t = (self.t-t_med)*u.day

        # model setup
        mask = np.isin(self.fid,filters)
        model = BoxLeastSquares(_t[mask],self.yn[mask], self.dyn[mask])

        # durations to search
        durmin = np.max([30./3600/24,self.p*0.005])
        durmax = np.min([10.,self.p*0.15])
        dur = np.logspace(np.log10(durmin),np.log10(durmax),num=5,endpoint=True)

        # run search
        out = model.autopower(np.array(dur),
            minimum_period = pmin,
            maximum_period = pmax)

        # select best period
        i = np.argmax(out.power)
        p = out.period[i].value
        t0 = out.transit_time[i].value + t_med

        print('period set to %12.12f, a %f fractional change' %(p,(self.p-p)/self.p))
        try:        
            print('t0 set to %g, %g fraction of the period' %(t0,(t0-self.t0)/p))
        except:
            pass        
        self.p = p
        self.t0 = t0
        self.dur = out.duration[i].value/p

        self.astropyBLS = out
        


    def refine_period_LS(self,filters=[1,2],logtrange=-5,Nharm=5):
        import astropy.units as u
        from astropy.timeseries import BoxLeastSquares
        from gatspy import periodic    

        # pmin,pmax
        pmin = self.p*(1-10**logtrange),
        pmax = self.p*(1+10**logtrange),

        # model setup

        model = periodic.LombScargleMultiband(fit_period=True)
        model.optimizer.period_range=(pmin, pmax)

        mask = np.isin(self.fid,filters)
        model.fit(self.t[mask], self.y[mask], self.dy[mask], self.fid[mask])


        p = model.best_period
        print('period set to %12.12f, a %f fractional change' %(p,(self.p-p)/self.p))
        self.p = p

    def plotBLS(self,n=10000,s=5):
        from scipy.special import ndtr

        # plot the BLS periodogram
        x,y = self.BLS['period'],self.BLS['power']

        # plot data
        plt.step(x,y)

        l = len(x)
        out = [[np.median(x[ndx:min(ndx + n, l)]),
                np.median(y[ndx:min(ndx + n, l)]),
                np.percentile(y[ndx:min(ndx + n, l)],ndtr(s)*100)] 
                    for ndx in range(0, l, n)]
        out = np.array(out)

        # plot median curve
        plt.plot(out[:,0],out[:,1],'C2-')
        plt.plot(out[:,0],out[:,2],'C3-')

        # layout
        plt.xlabel('period')
        plt.ylabel('power')
        plt.show()



    def plotlc(self,folded=False,mag=False,period=None,showflagged=True):
        # plot the lightcurve
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        colours = ['C2','C3','k']
        markers = ['o','o','x']

        for a,c in zip([1,2,3],colours): # loop over filters
            for phot,flagged,marker in zip([0,0,1],[True,False,True],markers):
                m = (self.fid==a)*(self.alert==phot)*((self.flag==0)==flagged)
                mfc="None" if not flagged else None
                
                # do not show flagged data
                if flagged>0 and not showflagged:
                    continue                

                if folded:
                    x = (self.t[m]-self.t0)/self.p%1
                    for n in [-1,0,1]:
                        ax1.errorbar(x+n,self.y[m]*10**6,self.dy[m]*10**6,
                            marker=marker,c=c,ls='none',markerfacecolor=mfc)

                else:
                    ax1.errorbar(self.t[m],self.y[m]*10**6,self.dy[m]*10**6,
                        marker=marker,c=c,ls='none',markerfacecolor=mfc)

        # 
        if folded:
            plt.xlim(-0.1,1.1)

        # add axis 2 labels
        ax2.set_ylim(ax1.get_ylim())
        minmag = -2.5*np.log10(np.max(ax1.get_ylim())/(3631.*10**6))
        maxmag = np.nanmin([23.01,-2.5*np.log10(np.min(ax1.get_ylim())/(3631.*10**6))])
        print(minmag)
        print(maxmag)
        maglabels = np.arange(np.ceil(minmag),np.floor(maxmag))[::-1]
        yticks = 3631*10**6*10**(-0.4*maglabels)
        ylabels = maglabels
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(ylabels)

        ax1.set_ylabel('muJy')
        ax2.set_ylabel('mag')
        plt.xlabel('phase')
        plt.show()








