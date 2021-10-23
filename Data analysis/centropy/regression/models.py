import numpy as np 

def linear(t,x):
    
    return x[0]*t + x[1]


def piecewise_linear(t,x):
    # x = [x0, y0, k1, k2]
    return np.piecewise(t, [t < x[0]], [lambda z: x[2]*z + x[1]-x[2]*x[0], lambda z:x[3]*z + x[1]-x[3]*x[0]])


def piecewise_linear_plateau(t,x):
    # x = [x0, y0, k1, k2]
    return np.piecewise(t, [t < x[0]], [lambda z: x[2]*z + x[1]-x[2]*x[0], lambda z:x[2]*x[0] + x[1]-x[2]*x[0]])
    

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def lorentzian(x,mu,sig):
    return .5*sig/((x-mu)**2 + (.5*sig)**2)
    

def multi_gaussian_peak_fn(t,x):
    fnc = 0
    n_gauss = len(x) // 3
    for i in range(n_gauss):
        fnc += x[3*i] * gaussian(t, x[3*i+1], x[3*i+2]) 
    return  fnc


def multi_lorentzian_peak_fn(t,x):
    fnc = 0
    n_comp = len(x) // 3
    for i in range(n_comp):
        fnc += x[3*i] * lorentzian(t, x[3*i+1], x[3*i+2]) 
    return  fnc


def gaussian_1D(depth_range, intensities, amplitude, mean, stddev):
    
    from astropy.modeling import models, fitting
    g_init = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)
    fit_g = fitting.LevMarLSQFitter()
    
    return fit_g(g_init, depth_range, intensities)


def lorentz_1D(depth_range, intensities, amplitude, mean, stddev):
    
    from astropy.modeling import models, fitting
    l_init = models.Lorentz1D(amplitude=amplitude, x_0=mean, fwhm=stddev)
    fit_l = fitting.LevMarLSQFitter()
    
    return fit_l(l_init, depth_range, intensities)