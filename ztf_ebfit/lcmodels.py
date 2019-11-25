import numpy as np
import ellc



def kepler_calc_RV(M1,M2,P,i):
    '''calculate the observed RV of both components

    input :
    M1 : float or array-like
        mass of star 1 [M_sun]
    M2 : float or array-like
        mass velocity of star 2 [M_sun]
    P : float or array-like
        orbital period [d]
    i : float or array-like
        inclination [degrees]

    output : tuple
        radial velocity 1 [km/s], radial velocity [km/s], semimajor-axis [R_sun]
    '''

    # constants in SI units
    G = 6.673*10**-11
    M_sun = 1.988*10**30
    R_sun = 6.955*10**8

    P = P*24*60*60 # convert from days to seconds
    i = np.radians(i) # convert to radians
    M1 = M1*M_sun # convert to SI
    M2 = M2*M_sun # convert to SI
    q = M2/M1

    # orbital separation
    a = (G*(M1+M2)/(4*np.pi**2)*P**2)**(1./3)

    # calculations
    K1 = 2*np.pi*a/P * q/(1+q) * np.sin(i)
    K2 = 2*np.pi*a/P * 1./(1+q) * np.sin(i)


    return K1*0.001,K2*0.001,a/R_sun



def WD_MR(M):
    '''calculate the radius of a zero temperature wd given a mass

    input :
    M : float or array-like
        the mass of the wd in solar units
    output : float or array-like
        the radius in solar units
    '''
    M_ch = 1.44
    M_p = 0.00057
    R = 0.0114 * (( M/M_ch )**(-2./3.)-(M/M_ch)**(2./3.))**(0.5) * (1+3.5*(M/M_p)**(-2./3.)+(M/M_p)**(-1))**(-2./3.)
    return R



def get_REg(q):
    """ the approximation by Eggleton for the volume radius of a roche lobe.

    Parameters
    ----------
    q : float
        the mass ratio of the binary

    Returns
    -------
    R : float
        the approximation to the roche lobe volumetric radius

    """
    R = 0.49*q**(2./3)
    R /= 0.6*q**(2./3) + np.log(1+q**(1./3.))
    return R







def ellc_WDRD(pars,x,logpars=False):

    # set values
    p,t0,r1,r2,incl,q,M1,scale,sbratio,heat = pars
    if logpars:
        sbratio = 10**sbratio

    # check parameter values
    if np.min(pars)<0:
        return -np.inf
    #print(p,t0,r1,r2,incl,q,sbratio,heat)
    if (r2 > 0.99*get_REg(q) or r1 > 0.99*get_REg(q**-1) or incl>=90 or 
            np.min([r1,r2,incl,q,scale,sbratio,heat])<=0):
        return -np.inf

    # calculate binary parameters
    K1,K2,a = kepler_calc_RV(M1,q*M1,p,incl)


    if M1>1.44 or r1*a<WD_MR(M1):
        print('WD mass/radius out of range')
        return -np.inf

    try:
        m = scale*ellc.lc(x,radius_1=r1,radius_2=r2,
            sbratio=sbratio,incl=incl,q=q,period=p,t_zero=t0,
            shape_2='roche',exact_grav=False, #WARNING TURN THIS ON IN PRODUCTION!!!
            ldc_1=0.6,ldc_2=0.6,gdc_2=0.08,heat_2=heat,
            grid_1='default',grid_2='default',
            t_exp=30./3600/24,n_int=3)
        #print('succes')
        return m
    except Exception as e:
        print('ellc failed:')
        print(e)
        return -np.inf




def ellc_WDWD(pars,x,logpars=False):

    # set values
    p,t0,r1,r2,incl,q,M1,scale,sbratio,heat = pars
    if logpars:
        sbratio = 10**sbratio

    # check parameter values
    if np.min(pars)<0:
        return -np.inf
    #print(p,t0,r1,r2,incl,q,sbratio,heat)
    if (r2 > 0.99*get_REg(q) or r1 > 0.99*get_REg(q**-1) or incl>=90 or 
            np.min([r1,r2,incl,q,scale,sbratio,heat])<=0):
        return -np.inf

    # calculate binary parameters
    K1,K2,a = kepler_calc_RV(M1,q*M1,p,incl)

    # lower limit to WD radius
    if M1>1.44 or r1*a<WD_MR(M1):
        print('WD mass/radius out of range')
        return -np.inf
    if M2>1.44 or r2*a<WD_MR(M2):
        print('WD mass/radius out of range')
        return -np.inf

    try:
        m = scale*ellc.lc(x,radius_1=r1,radius_2=r2,
            sbratio=sbratio,incl=incl,q=q,period=p,t_zero=t0,
            shape_2='roche',exact_grav=False, #WARNING TURN THIS ON IN PRODUCTION!!!
            ldc_1=0.6,ldc_2=0.6,gdc_2=0.08,heat_2=heat,
            grid_1='default',grid_2='default',
            t_exp=30./3600/24,n_int=3)
        #print('succes')
        return m
    except Exception as e:
        print('ellc failed:')
        print(e)
        return -np.inf






def ellc_WDRD_multiband(pars,t,f,filters=[1,2]):
    """ a multiband EB lightcurve model.

    input
    pars : array
        list of parameters;
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
    basepars = pars[:7] # p,t0,r1,r2,incl,q,M1

    for k,fid in enumerate(filters): # for each band, use EBmodel
        filtpars = pars[7+k*4:7+(k+1)*4]
        out = ellc_WDRD(np.r_[basepars,filtpars],t[f==fid])
        if np.isinf(out):
            return -np.inf
        else:
            y[f==fid] = out

    return y

