import numpy as np
import matplotlib.pyplot as plt
import periodfind

class Ztflc:
    """ ZTF lightcurve. The main object is lc, which is an array with
    t,flux,error,filter,flags,alerts
    """

    def __init__(self):
        pass # do nothing

    def load_fromfile(self,filename):
        self.lc = np.loadtxt(filename)

    def load_fromarray(self,array):
        self.lc = array

    def run_BLS(pmin=0.01,pmax=5.0,
            filters=[1,2],clean=True,alerts=True):
        """find period"""

        # select data
        m = np.isin(lc[:,3],filters) # select data on filters
        if clean: # use only clean data
            m *= lc[:,4] == 0
        if not alerts: # use only PSF phot
            m *= lc[:,5] == 0

        p,t0,q,period,power = run_BLScuvarbase(lc,pmin,pmax,oversampling=3
                        qmin=0.01,qmax=0.1,dlogq=0.1)
        
        self.p = p
        self.t0 = t0
        self.q = q
        self.BLS_period = period
        self.BLS_power = power




    def plotlc(self,folded=False,mag=False,period=None):
        # plot the lightcurve
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()


        colours = ['C2','C3','k']
        markers = ['o','o','x']

        for a,c in zip([1,2,3],colours): # loop over filters
            for phot,flagged,marker in zip([0,0,1],[True,False,True],markers):
                m = (self.lc[:,3]==a)*(self.lc[:,5]==phot)*((self.lc[:,4]==0)==flagged)
                mfc="None" if not flagged else None
                if folded:
                    x = (self.lc[m,0]-self.t0)/self.p%1
                    ax1.errorbar(x,self.lc[m,1]*10**6,self.lc[m,2]*10**6,
                        marker=marker,c=c,ls='none',markerfacecolor=mfc)
                else:
                    ax1.errorbar(self.lc[m,0],self.lc[m,1]*10**6,self.lc[m,2]*10**6,
                        marker=marker,c=c,ls='none',markerfacecolor=mfc)

        # add axis 2 labels
        ax2.set_ylim(ax1.get_ylim())
        minmag = -2.5*np.log10(np.max(ax1.get_ylim())/3631.)
        maxmag = np.min([22.01,-2.5*np.log10(np.min(ax1.get_ylim())/3631.)])
        maglabels = np.arange(np.ceil(18.3),np.floor(22.1))
        yticks = 3631*10**6*10**(-0.4*maglabels)
        ylabels = maglabels
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(ylabels)

        ax1.set_ylabel('muJy')
        ax2.set_ylabel('mag')
        plt.xlabel('phase')
        plt.show()
