import numpy as np

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


