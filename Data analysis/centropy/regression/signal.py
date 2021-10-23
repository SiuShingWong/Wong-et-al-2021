
import numpy as np 
import centropy.regression.filters as filters
import centropy.regression.fitting as fitting
from centropy.regression.utils import *

class Signal():
    def __init__(self, x, y, kind='linear'):
        self.x = x
        self.y = y

        self.dx = np.diff(self.x)[0]

        self.x, self.y = interp_signal(self.x, self.y, kind=kind)
        self.y_norm, self.norm_params = min_max_normalise(self.y)

class LinearSignal(Signal):
    def fit(self, p0_linear=None, fit_loss='soft_l1'):
        y_target = self.y_norm

        self.linear_params, self.linear_model = fitting.robust_fit_linear(self.x, y_target, p0=p0_linear, f_scale=0.05, max_nfev=None, fit_loss=fit_loss)

        params_robust_linear = self.linear_params.x

        params_robust_linear[0] = params_robust_linear[0] * (self.norm_params[1]-self.norm_params[0])
        params_robust_linear[1] = min_max_denormalise(params_robust_linear[1], self.norm_params[0], self.norm_params[1])

        self.linear_y = self.linear_model(self.x, params_robust_linear)

        self.regress_model = self.linear_model
        self.regress_params = params_robust_linear
        self.regress_y = self.linear_y
        self.cost = self.linear_params.cost

class LinearPlateauSignal(Signal):
    def fit(self, p0_piecewise=None, fit_loss='soft_l1'):
        y_target = self.y_norm

        self.linear_plateau_params, self.linear_plateau_model = fitting.robust_fit_linear_plateau(self.x, y_target,
                                                                   p0=p0_piecewise, f_scale=0.05, max_nfev=None, fit_loss=fit_loss)

        params_robust_linear_plateau = self.linear_plateau_params.x

        params_robust_linear_plateau[1] = min_max_denormalise(params_robust_linear_plateau[1], self.norm_params[0], self.norm_params[1])
        params_robust_linear_plateau[2:] = params_robust_linear_plateau[2:] * (self.norm_params[1]-self.norm_params[0])

        self.linear_plateau_y = self.linear_plateau_model(self.x, params_robust_linear_plateau)

        self.regress_model = self.linear_plateau_model
        self.regress_params = params_robust_linear_plateau
        self.regress_y = self.linear_plateau_y
        self.cost = self.linear_plateau_params.cost

class PiecewiseLinearSignal(Signal):
    def fit(self, p0_piecewise=None, fit_loss='soft_l1'):
        y_target = self.y_norm

        self.piecewise_params, self.piecewise_linear_model = fitting.robust_fit_piecewise(self.x, y_target,
                                                                   p0=p0_piecewise, f_scale=0.05, max_nfev=None, fit_loss=fit_loss)

        params_robust_piecewise = self.piecewise_params.x

        params_robust_piecewise[1] = min_max_denormalise(params_robust_piecewise[1], self.norm_params[0], self.norm_params[1])
        params_robust_piecewise[2:] = params_robust_piecewise[2:] * (self.norm_params[1]-self.norm_params[0])

        self.piecewise_y = self.piecewise_linear_model(self.x, params_robust_piecewise)

        self.regress_model = self.piecewise_linear_model
        self.regress_params = params_robust_piecewise
        self.regress_y = self.piecewise_y
        self.cost = self.piecewise_params.cost

class CycleSignal():
    def __init__(self, x, y, single_cycle): 
        self.x = x
        self.y = y
        self.single_cycle = single_cycle

    # create and fit a multi-gaussian function.
    def fit_multi_gauss(self, refine_trough=True, 
                        refine_peak_trough=True, 
                        baseline='linear',
                        smooth_method='GP', 
                        smoothing=.75, # Originally 0.75
                        smooth_border=None,
                        peak_border='reflect',
                        peak_trough_prominence=10, # Modify this for less normalized or raw data (Absolute number)
                        fit_loss='cauchy'):
        """
        To Do: expose adjustable parameters of the different steps.
        """
        # 1. detrend
        self.detrend(baseline=baseline)

        # 2. smooth
        self.smooth(method=smooth_method, border=smooth_border, smoothing=smoothing)

                
        # 3. find peaks and troughs
        if self.single_cycle == True:
            self.find_signal_peaks_and_troughs(border=None, prominence=peak_trough_prominence)        
        else:
            self.find_signal_peaks_and_troughs(border=peak_border, prominence=peak_trough_prominence)  
        
        # 4. define multi-gauss fitting.
        self.multi_gauss_model = fitting.Multi_Gauss_Signal(self.x, self.y_smooth, 
                                                    self.peak_index,
                                                    troughs=self.trough_index,
                                                    fit_loss=fit_loss)
        # fit peaks
        self.multi_gauss_model.fit(f_scale=0.05)
        
        if refine_trough:
            # refine fit for troughs
            self.multi_gauss_model.refine_fit_troughs()
        
        if refine_peak_trough:
            # update the description.
            self.find_signal_peaks_and_troughs(y=self.multi_gauss_model.sample(self.x),  border=None, remove_mean=True, min_peak_sep=5, prominence=peak_trough_prominence, normalize=False)
            
        # reconstruct the full signal fit for easy use.
        self.gauss_fit_signal = self.multi_gauss_model.sample(self.x) + self.x*self.detrend_params[0]

        self.cost = self.multi_gauss_model.fit_params.cost
            
    def extract_peak_increase(self, use_gauss_fit=True, y=None, direction=None):
        
        self.peak_increase = self.extract_slopes(use_gauss_fit=use_gauss_fit, y=y, direction='increase')
        
    def extract_peak_decrease(self, use_gauss_fit=True, y=None, direction=None):
        
        self.peak_decrease = self.extract_slopes(use_gauss_fit=use_gauss_fit, y=y, direction='decrease')
        
    def find_fwhm_peaks_troughs(self, samples=100):
        
        # iterate through each peaks and troughs to extract this. 
        peak_pos = self.peak_index
        trough_pos = self.trough_index
        x_fine = np.linspace(self.x[0], self.x[-1], 10*samples)
        
        # grab the global turning points as a more accurate upper lower.
        grad_gauss = np.diff(self.multi_gauss_model.sample(time=x_fine))
        grad_crossing = np.where(np.diff(np.sign(grad_gauss-0)))[0]
        grad_time = x_fine[grad_crossing]
        
        fwhm_peaks = []
        fwhm_troughs = []

        if len(grad_time) == 0:
            fwhm_peaks = []
            fwhm_troughs = []
        else:
            
            for i in range(len(peak_pos)):
                pos = peak_pos[i]
                lower = trough_pos[trough_pos < pos]
                upper = trough_pos[trough_pos > pos]
    
                # sort out the lower and upper bounds.        
                if len(lower) > 0:
                    lower = self.x[lower[-1]]
                else:
                    lower = self.x[0] # first time point
                if len(upper) > 0:
                    upper = self.x[upper[0]]
                    upper = grad_time[np.argmin(np.abs(grad_time-upper))]
                else:
                    upper = self.x[-1] # last time point
            
                if lower == upper:
                    FWHM = [[], np.nan]
                else:
                    
                    lower_nearest = grad_time[np.argmin(np.abs(grad_time-lower))]
                    upper_nearest = grad_time[np.argmin(np.abs(grad_time-upper))]
                    # check these are not returning the query min/max.             
                    if np.abs(lower_nearest - self.x[peak_pos][0]) > 1:
                        lower = lower_nearest
                    if np.abs(upper_nearest - self.x[peak_pos][0]) > 1:
                        upper = upper_nearest
                                              
                    x_fine = np.linspace(lower, upper, samples) # sampling resolution
                    y_fine = self.multi_gauss_model.sample(time=x_fine)
                    
                    y_lower = np.maximum(y_fine[0], y_fine[-1]) # which is higher of the bounds
                    y_peak = np.max(y_fine)
                    half_max = .5*(y_lower + y_peak) # half way.
                    zero_crossing = np.where(np.diff(np.sign(y_fine-half_max)))[0]
                    zero_time = x_fine[zero_crossing]
        
                    # gives FWHM 
                    if len(zero_time) >= 2:
                        FWHM = np.diff(zero_time)[0]
                        # print(zero_time)
                        FWHM = [[zero_time[0], zero_time[1]], FWHM] # also give the range.
                    else:
                        FWHM = [[], np.nan]
                fwhm_peaks.append(FWHM)
                
                
            for i in range(len(trough_pos)):
                pos = trough_pos[i]
                lower = peak_pos[peak_pos < pos]
                upper = peak_pos[peak_pos > pos]
    
                # sort out the lower and upper bounds.        
                if len(lower) > 0:
                    lower = self.x[lower[-1]]
                else:
                    lower = self.x[0] # first time point
                if len(upper) > 0:
                    upper = self.x[upper[0]]
                else:
                    upper = self.x[-1] # last time point
                
                # to do: refine this by finding the points closes to zero gradient.
    
                if lower == upper:
                    FWHM = [[],np.nan]
                else:
                    lower_nearest = grad_time[np.argmin(np.abs(grad_time-lower))]
                    upper_nearest = grad_time[np.argmin(np.abs(grad_time-upper))]
                    # check these are not returning the query min/max.             
                    if np.abs(lower_nearest - self.x[peak_pos][0]) > 1:
                        lower = lower_nearest
                    if np.abs(upper_nearest - self.x[peak_pos][0]) > 1:
                        upper = upper_nearest
                     
                    x_fine = np.linspace(lower, upper, samples) # sampling resolution
                    y_fine = self.multi_gauss_model.sample(time=x_fine)
                    
                    y_upper = np.minimum(y_fine[0], y_fine[-1]) # which is higher of the bounds
                    y_trough = np.min(y_fine)
        
                    half_max = .5*(y_upper + y_trough) # half way.
                    zero_crossing = np.where(np.diff(np.sign(y_fine-half_max)))[0]
                    zero_time = x_fine[zero_crossing]
        
                    # gives FWHM 
#                    FWHM = np.diff(zero_time)[0]
#                    FWHM = [[zero_time[0], zero_time[1]], FWHM]
                    if len(zero_time) >=2:
                        FWHM = np.diff(zero_time)[0]
                        FWHM = [[zero_time[0], zero_time[1]], FWHM] # also give the range.
                    else:
                        FWHM = [[], np.nan]
                fwhm_troughs.append(FWHM)    
                
        self.fwhm_peaks = fwhm_peaks
        self.fwhm_troughs = fwhm_troughs
     
    """
    define helper functions.
    """ 
    def detrend(self, baseline='linear'):
        self.y_detrend, self.detrend_params = filters.detrend_signal(self.x, self.y, 
                                                                      lam=1e6, p=0.1, niter=10, 
                                                                      border=None, 
                                                                      sides='both', method='linear')

    def smooth(self, y=None, smoothing=0.75, cycle=None, method='GP', border='reflect', sides='both', lam=0.1, p=0.5, niter=30):
        if y is None:
            y_in = self.y_detrend
        else:
            y_in = y.copy()
            
        if method == 'GP':
            self.y_smooth = filters.smooth_signal_GP(y_in, smoothing=smoothing)
        if method == 'ls':
            self.y_smooth = filters.baseline_als(y_in, border=border, sides=sides, 
                                              lam=lam, p=p, niter=niter)            
          
    def find_signal_peaks_and_troughs(self, y=None,  border='reflect', remove_mean=True, min_peak_sep=5, prominence=None, normalize=False):
        """
        TO DO: check performance of this function. (in terms of peak prominence.)
        """
        if y is None:
            y_in = self.y_smooth
        else:
            y_in = y.copy()
            
        if prominence is None:
            prominence = 1*np.std(y_in)/2.
            
        self.peak_index = detect_peaks(y_in, border=border, remove_mean=remove_mean, min_peak_sep=min_peak_sep, prominence=prominence)
        self.trough_index = detect_peaks(y_in.max()-y_in, border=border, remove_mean=remove_mean, min_peak_sep=min_peak_sep, prominence=prominence)
        # print (self.peak_index)
    

    def extract_slopes(self, use_gauss_fit=True, y=None, direction=None):
        
        # results accumulator.
        rates = []
        rate_poses = []
        
        if y is None:
            if use_gauss_fit:
                y = self.gauss_fit_signal 
            else:
                y = self.y_smooth # use the smooth version.
        
        if use_gauss_fit:
            peak_pos = self.multi_gauss_model.peak_index
            trough_pos = self.multi_gauss_model.trough_index
        else:
            peak_pos = self.peak_index
            trough_pos = self.trough_index
        
        if direction=='increase':
            n_i = len(peak_pos)
        if direction == 'decrease':
            n_i = len(trough_pos)
            
        # print(n_i, direction, len(trough_pos))
        # print(peak_pos)
        
        for i in range(n_i): # compute for as many as these as possible. 
            if direction=='increase':
                upper = peak_pos[i]
            if direction == 'decrease':
                upper = trough_pos[i]

            # this is the problematic line.
            if direction == 'increase':
                if np.sum(trough_pos < upper) > 0:
                    lower = trough_pos[trough_pos<upper][-1] # take the last one that is smaller. 
                else:
                    lower = 0     
            if direction == 'decrease':
                if np.sum(peak_pos < upper) > 0:
                    lower = peak_pos[peak_pos<upper][-1] # take the last one that is smaller. 
                else:   
                    lower = 0
                
            # print(direction, lower, upper)

            # isolate the range.
            select_range = np.arange(lower, upper+1, dtype=np.int)
            
            if len(select_range) > 1: # has to have 2 numbers.
#                time_range = time[select_range]
                signal_range = y[select_range]
                diff_range = np.diff(signal_range)
                diff_max_pos = np.argmax(np.abs(diff_range))
                
                # reference to absolute time.
                rate_pos = diff_max_pos + lower # return the index
                rate = diff_range[diff_max_pos]
                rates.append(rate)
                rate_poses.append(rate_pos)
            else:
                rate = np.nan
                rate_pos = np.nan
                rates.append(rate)
                rate_poses.append(rate_pos)
            
        if len(rates) > 0:
            rates = np.hstack(rates)
        if len(rate_poses) > 0:
            rate_poses = np.hstack(rate_poses).astype(np.int)
            
        return rate_poses, rates






