import numpy as np
from astropy.time import Time
from astropy import coordinates as coord
from astropy import units as u

def print_coords(ra,dec):
    """ print coordinates in hhmmss+ddmmss """
    c = coord.SkyCoord(ra=ra*u.deg,dec=dec*u.deg, frame='icrs')
    txt = c.to_string('hmsdms')
    txt = txt.replace("h",":")
    txt = txt.replace("m",":")
    txt = txt.replace("s","")
    txt = txt.replace("d",":")
    print(txt)


def nearest_eclipse(t,p,t0):
    N = np.round((t-t0)/p)
    tn = t0 + N*p
    return tn

def mag2flux(mag,dmag=[],flux_0 = 3631.0):
    # converts magnitude to flux in Jy
    flux = flux_0 * 10**(-0.4*mag)

    if dmag==[]:
        return flux
    else:
        dflux_p = (flux_0 * 10**(-0.4*(mag-dmag)) - flux)
        dflux_n = (flux_0 * 10**(-0.4*(mag+dmag)) - flux)
        return flux, dflux_p, dflux_n

def flux2mag(flux,dflux=[],flux_0 = 3631.0):
    # converts flux (Jy) to magnitudes
    mag = -2.5*np.log10(flux/flux_0)

    if dflux == []:
        return mag

    else:
        dmag_p = -2.5*np.log10((flux-dflux)/flux_0) - mag
        dmag_n = -2.5*np.log10((flux+dflux)/flux_0) - mag

        return mag, dmag_p, dmag_n

def JD2HJD(JD,ra,dec,site='palomar'):
    """ convert a JD time in a HJD time in utc"""
    target = coord.SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    tsite = coord.EarthLocation.of_site(site)
    times = Time(JD, format='jd',
                      scale='utc', location=tsite)
    ltt_helio = times.light_travel_time(target, 'heliocentric')

    HJD = times.utc+ltt_helio

    return HJD.jd

def JD2BJD(JD,ra,dec,site='palomar'):
    """ convert a JD time in a HJD time in tdb"""
    target = coord.SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    tsite = coord.EarthLocation.of_site(site)
    times = Time(JD, format='jd',
                      scale='utc', location=tsite)
    ltt_bary = times.light_travel_time(target, 'barycentric')

    BJD = times.tdb+ltt_bary

    return BJD.jd

def HJD2JD(HJD,ra,dec,site='palomar'):
    """ convert a HJD time (utc) in a JD time in utc. Note that this correction is 
    not exact because the time delay is calculated at the HJD time, and not the 
    JD time. The difference is very small however, and should not be a problem 
    for 0.01sec timings."""
    target = coord.SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    tsite = coord.EarthLocation.of_site(site)
    times = Time(HJD, format='jd',
                      scale='utc', location=tsite)
    ltt_helio = times.light_travel_time(target, 'heliocentric')

    JD = times.utc-ltt_helio

    return JD.jd

def BJD2JD(BJD,ra,dec,site='palomar'):
    """ convert a BJD time (tdb) in a JD time in utc. Note that this correction 
    is not exact because the time delay is calculated at the HJD time, and not 
    the JD time. The difference is very small however, and should not be a 
    problem for 0.01sec timings."""
    target = coord.SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    tsite = coord.EarthLocation.of_site(site)
    times = Time(BJD, format='jd',
                      scale='tdb', location=tsite)
    ltt_bary = times.light_travel_time(target, 'barycentric')

    JD = times.utc-ltt_bary

    return JD.jd

def HJD2BJD(HJD,ra,dec,site='palomar'):
    """ convert a HJD time (utc) in a BJD time in tdb. Note that this correction 
    is not exact because the time delay is calculated at the HJD time, and not 
    the JD time. The difference is very small however, and should not be a 
    problem for 0.01sec timings."""
    target = coord.SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    tsite = coord.EarthLocation.of_site(site)
    times = Time(HJD, format='jd',
                      scale='utc', location=tsite)

    # remove heliocentric correction
    ltt_helio = times.light_travel_time(target, 'heliocentric')
    JD = (times.utc - ltt_helio).jd # remove heliocentric correction

    JDtimes = Time(JD, format='jd',
                      scale='utc', location=tsite)

    # do barycentric correction
    ltt_bary = JDtimes.light_travel_time(target, 'barycentric')

    BJD = JDtimes.tdb+ltt_bary

    return BJD.jd



