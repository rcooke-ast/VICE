from scipy import interpolate
import numpy as np
import vice
from vice.toolkit.interpolation import interp_scheme_2d
from IPython import embed

yieldsdir = "/Users/rcooke/Work/Research/BBN/helium34/GCE/yields/"

# Setup AGB yields
mass3, Zmet3, yldpm3 = np.loadtxt(yieldsdir + "Lagarde2011_3He_yields.csv", delimiter=",", unpack=True, usecols=(0,1,3))
mass4, Zmet4, yldpm4 = np.loadtxt(yieldsdir + "Lagarde2011_4He_yields.csv", delimiter=",", unpack=True, usecols=(0,1,3))
sh = (4,9,)
metval = Zmet3.reshape(sh)[:,0]
# Add a value for 8 Msun
massval = np.append(mass3.reshape(sh)[0,:], 8.0)
yld3 = np.append(yldpm3.reshape(sh).T, -1.75E-5*np.ones((1,metval.size)), axis=0)
yld4 = np.append(yldpm4.reshape(sh).T, yldpm4.reshape(sh).T[-1,:].reshape((1,metval.size)), axis=0)

yld3spl_agb = interp_scheme_2d(massval, metval, yld3)
yld4spl_agb = interp_scheme_2d(massval, metval, yld4)
#yld3spl_agb = interpolate.RectBivariateSpline(massval, metval, yld3, kx=1, ky=1)
#yld4spl_agb = interpolate.RectBivariateSpline(massval, metval, yld4, kx=1, ky=1)
# Don't add a value for 8 Msun
# massval = mass3.reshape(sh)[0,:]
# yld3spl_agb = interpolate.RectBivariateSpline(massval, metval, yldpm3.reshape(sh).T, kx=1, ky=1)
# yld4spl_agb = interpolate.RectBivariateSpline(massval, metval, yldpm4.reshape(sh).T, kx=1, ky=1)

# Setup CCSNe yields
data3 = np.load(yieldsdir + "LC2018_3He_yields_Kroupa.npy")
data4 = np.load(yieldsdir + "LC2018_4He_yields_Kroupa.npy")
#data3 = np.load(yieldsdir + "CL2004_3He_yields_Scalo86.npy")
#data4 = np.load(yieldsdir + "CL2004_4He_yields_Scalo86.npy")
yld3spl_ccsne = interpolate.interp1d(data3[:,0], data3[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
yld4spl_ccsne = interpolate.interp1d(data4[:,0], data4[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')


def custom_agb_yield_3He(mass, z):
    # Mass and metallicity dependent model
    return yld3spl_agb(mass, z)


def custom_agb_yield_4He(mass, z):
    # Mass and metallicity dependent model
    return yld4spl_agb(mass, z)


def custom_ccsne_yield_3He(zval):
    # Metallicity dependent model
    return yld3spl_ccsne([zval])[0]


def custom_ccsne_yield_4He(zval):
    # Metallicity dependent model
    return yld4spl_ccsne([zval])[0]


# Change some of the yield settings
vice.yields.agb.settings['he'] = custom_agb_yield_4He
vice.yields.agb.settings['au'] = custom_agb_yield_3He
vice.yields.ccsne.settings['he'] = custom_ccsne_yield_4He
vice.yields.ccsne.settings['au'] = custom_ccsne_yield_3He
vice.yields.sneia.settings['au'] = 0

# Set a metallicity dependent mass-lifetime relation (MLR)
#vice.mlr.setting = "hpt2000"
