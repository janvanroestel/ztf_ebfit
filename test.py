import ztf_ebfit
import numpy as np

data = np.loadtxt('data/lc_249.4316_49.2947.dat')


mylc = ztf_ebfit.Ztflc()
mylc.load_fromarray(data[:,[0,1,2,3,7,8]])
mylc.plotlc()
