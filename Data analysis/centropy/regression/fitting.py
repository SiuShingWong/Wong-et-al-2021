
import numpy as np
import centropy.regression.models as models
from scipy.optimize import least_squares
from centropy.regression.utils import *

def robust_fit_piecewise(t,y, p0=None, f_scale=0.001, max_nfev=None, fit_loss='huber'):
    
#    def piecewise_linear(t,x):
#        # x = [x0, y0, k1, k2]
#        return np.piecewise(t, [t < x[0]], [lambda z: x[2]*z + x[1]-x[2]*x[0], lambda z:x[3]*z + x[1]-x[3]*x[0]])
    
    def residual(x, t, y):
        return (models.piecewise_linear(t,x) - y) ** 2
    
    # f_scale may need tuning. 
    bounds = ([0, 0, 0, -np.inf], [t.max(), y.max(), np.inf, 0])
    res_robust = least_squares(residual, p0, loss=fit_loss, f_scale=f_scale, args=(t, y), max_nfev=max_nfev, bounds=bounds) # whats the best loss?
   
    return res_robust, models.piecewise_linear

def robust_fit_linear(t,y, p0=None, f_scale=0.001, max_nfev=None, fit_loss='huber'):
    
#    def linear(t,x):
#        # x = [x0, y0, k1, k2]
#        return x[0]*t + x[1]
#    
    def residual(x, t, y):
        return (models.linear(t,x) - y) ** 2
    
    # f_scale may need tuning. 
    res_robust = least_squares(residual, p0, loss=fit_loss, f_scale=f_scale, args=(t, y), max_nfev=max_nfev) # whats the best loss?
   
    return res_robust, models.linear
    
def robust_fit_linear_plateau(t,y, p0=None, f_scale=0.001, max_nfev=None, fit_loss='huber'):
    
# def piecewise_linear_plateau(t,x):
#     # x = [x0, y0, k1, k2]
#     return np.piecewise(t, [t < x[0]], [lambda z: x[2]*z + x[1]-x[2]*x[0], lambda z:x[2]*z])  
    def residual(x, t, y):
        return (models.piecewise_linear_plateau(t,x) - y) ** 2
    
    # f_scale may need tuning. 
    res_robust = least_squares(residual, p0, loss=fit_loss, f_scale=f_scale, args=(t, y), max_nfev=max_nfev) # whats the best loss?
   
    return res_robust, models.piecewise_linear_plateau

class Multi_Gauss_Signal():
    
    def __init__(self, x, y, peaks, troughs=None, fit_loss='cauchy'): 
        self.x = x 
        self.y = y
        self.dx = np.diff(x)[0]

        # normalize signal.
        self.y_norm, self.norm_params = min_max_normalise(self.y)

        self.peak_index = peaks.astype(np.int) # cast into int
        self.peak_times = self.x[self.peak_index]
        self.peak_amp = self.y[self.peak_index]

        self.trough_index = troughs.astype(np.int) # cast into int
        self.trough_times = self.x[self.trough_index]
        self.trough_amp = self.y[self.trough_index]

        self.all_index = np.sort(np.hstack([self.peak_index, self.trough_index]))
        self.all_times = self.x[self.all_index]
        self.all_amp = self.y[self.all_index]

        self.fit_loss = fit_loss

    def fit(self, normalize=True, peak_pos=None, p0=None, s0=None, f_scale=0.001, max_nfev=None):
        """
        Only makes sense for peak fits.
        """
        from scipy.optimize import least_squares
        
        x_in = self.x
        self.fit_peak_normalize = normalize # save this setting. (required for recovering original)
        
        if peak_pos is None:
            peak_pos = self.peak_times
           
        if normalize:
            peak_amp = self.y_norm[self.peak_index] # use the normalised to fit?
            y_in = self.y_norm.copy()
        else:
            peak_amp = self.peak_amp
            y_in = self.y
#        peak_std = np.hstack([np.mean(np.diff(self.peak_times)) / 2.] * len(peak_pos))
        peak_std = np.hstack([1.] * len(peak_pos))
        
        n_gauss = len(peak_pos)
        if p0 is None:
            x0 = np.hstack([np.array([peak_amp[i], peak_pos[i], peak_std[i]]) for i in range(n_gauss)])
        else:
            x0 = np.hstack([np.array([p0[i], peak_pos[i], s0[i]]) for i in range(n_gauss)])
         
            
        def residual(x, t, y):
            return (models.multi_gaussian_peak_fn(t,x) - y) ** 2
            
        # robust least squares fit.
        self.fit_params = least_squares(residual, x0, loss=self.fit_loss, 
                                   f_scale=f_scale, 
                                   args=(x_in, y_in), max_nfev=max_nfev) 
                
    def refine_fit_troughs(self, normalize=True, p0=-0.1, s0=10., f_scale=0.001, max_nfev=None):
        """
        enables more troughs to be added ad-hoc to an existing fit.
        """
        p0_params = self.fit_params.x[::3]
        s0_params = self.fit_params.x[2::3]
        pos_params = self.fit_params.x[1::3]
#        # redetect peaks ?
#        peak_pred = self.sample()
#        peaks_fit = sig.detect_peaks(peak_pred, border=None, remove_mean=True, min_peak_sep=5, prominence=None)
        
        # add in the troughs.
        p = np.hstack([p0_params, p0*np.ones(len(self.trough_times))])
        s = np.hstack([s0_params, s0*np.ones(len(self.trough_times))])
        peaks = np.hstack([pos_params, self.trough_times])
        
        # recall the fit function.
        self.fit(normalize=normalize, peak_pos=peaks, p0=p, s0=s, f_scale=f_scale, max_nfev=None)
        
    def sample(self, time=None, params=None):
        
        if time is None and params is None:
            y_pred = models.multi_gaussian_peak_fn(self.x, self.fit_params.x)
        if time is None and params is not None:
            y_pred = models.multi_gaussian_peak_fn(self.x, params)
        if time is not None and params is not None:
            y_pred = models.multi_gaussian_peak_fn(time, params)
        if time is not None and params is None:
            y_pred = models.multi_gaussian_peak_fn(time, self.fit_params.x)
        
        if self.fit_peak_normalize == True:
            # renormalize.
            y_pred = min_max_denormalise( y_pred, self.norm_params[0], self.norm_params[1])

        return y_pred
        
    def residual(self, x, t, y):
        return (models.multi_gaussian_peak_fn(t,x) - y) ** 2
    
