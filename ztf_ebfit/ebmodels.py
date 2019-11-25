def EBmodel(pars,t):
    """ a simple EB lightcurve model.

    input
    pars : array
        list of parameters; period, t0, width 1, width 2, scale, 
            depth 1, depht 2, refl, ell
    t : array
        times of binary lightcurve

    output:
        lightcurve values in flux
    """

    # extract parameters
    p,t0,w1,w2,scale,e1,e2,refl,ell = pars

    # convert some parameters to log
    y = np.ones_like(t)
    x = (t-t0)/p%1 # calculate phases

    # eclipses
    m1 = (x<w1)|(x>1.-w1)
    y[m1] -= e1

    # egress eclipse 1
    m2 = (x>w1)&(x<w1+w2)
    r1 = e1/w2
    y[m2] = r1*(x[m2]-w1)+(1-e1)

    # ingress eclipse 1
    xp = abs(1-x)
    m2 = (xp>w1)&(xp<w1+w2)
    r1 = e1/w2
    y[m2] = r1*(xp[m2]-w1)+(1-e1)

    # if eclipse depth 2 is not zero, add eclipse 2
    if not e2==0:
        # eclipses
        m = (x<0.5+w1)&(x>0.5-w1)
        y[m] -= e2

        # egress eclipse 2
        m2 = (x>w1+0.5)&(x<w1+w2+0.5)
        r2 = e2/w2
        y[m2] = r2*(x[m2]-0.5-w1)+(1-e2)

        # egress eclipse 2
        m2 = (x<0.5-w1)&(x>0.5-w1-w2)
        r2 = e2/w2
        y[m2] = -r2*(x[m2]-0.5+w1+w2)+1

    # sinusoidal components (reflection, ellipsoidal)
    y += refl*0.5*(1-np.cos(2*np.pi*x))
    y += ell*0.5*(1-np.cos(4*np.pi*x))

    # rescale model
    y *= scale

    return y




def EBmodel_multiband(pars,t,f,filters=[1,2]):
    """ a multiband EB lightcurve model.

    input
    pars : array
        list of parameters; phi0, eclipse width 1, eclipse width 2, 
            scaling, eclipse depth1, eclipse depth2, refl, ellipsoidal
    t : array
        times of binary lightcurve
    f : array
        indicates which band the observations was taking in (array of ints)

    output:
        lightcurve values in flux
    """

    # make emptly LC
    y = np.ones_like(t)

    # prep parameters
    basepars = pars[:4] # p,t0,w1,w2
    for k,fid in enumerate(filters): # for each band, use EBmodel
        filtpars = pars[4+k*5:4+(k+1)*5]
        y[f==fid] = EBmodel(np.r_[basepars,filtpars],t[f==fid])
    
    return y



def trapezoid_multiband(pars,t,f,filters=[1,2]):
    """ a multiband EB lightcurve model.

    input
    pars : array
        list of parameters; phi0, eclipse width 1, eclipse width 2, 
            scaling, eclipse depth1, eclipse depth2, refl, ellipsoidal
    t : array
        times of binary lightcurve
    f : array
        indicates which band the observations was taking in (array of ints)

    output:
        lightcurve values in flux
    """

    # make emptly LC
    y = np.ones_like(t)

    # prep parameters
    basepars = pars[:4] # p,t0,w1,w2
    for k,fid in enumerate(filters): # for each band, use EBmodel
        filtpars = pars[4+k*5:4+(k+1)*5]
        y[f==fid] = EBmodel(np.r_[basepars,filtpars],t[f==fid])
    
    return y

