import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def min_max_normalise(y):
        
        min_y = np.min(y)
        max_y = np.max(y)
        y_ = (y - min_y) / float(max_y - min_y)
        
        return y_, (min_y, max_y)
    
def min_max_denormalise( y, min_sig, max_sig):
    
    y_ = y * (max_sig - min_sig) + min_sig

    return y_
        
    
# 1D signal interpolation for simple imputing of missing values
def interp_signal(x,y, kind='cubic'):
    """
    make dense the signal
    """
    
    val = np.logical_and(np.logical_not(np.isnan(y)), np.logical_not(np.isnan(x)))
    x_ = x[val].copy()
    y_ = y[val].copy()
    
    f = interp1d(x_,y_, kind=kind)
    
    # prevent extrapolation. (only interpolation)
    min_x = np.min(x_); max_x = np.max(x_)
    select = np.logical_and(x>=min_x, x<=max_x)
    
    return x[select], f(x[select])
    
    
def detect_peaks(y, border='reflect', remove_mean=True, min_peak_sep=5, prominence=None):
    
    y_ = y.copy()
    
    if remove_mean:
        y_ = y_ - np.mean(y_)
    
    if border == 'reflect':
        y_ = np.hstack([y_[::-1], y_])
    
    peaks = find_peaks(y_, prominence=prominence)
    
    if border == 'reflect':
        peak_pos = peaks[0] - len(y) # subtract.
        # check positions. 
        peak_pos = np.clip(peak_pos, 0, len(y))
        peak_pos = np.unique(peak_pos)
        
        if peak_pos[1] - peak_pos[0] < min_peak_sep:
            peak_pos = peak_pos[1:]
    else:
        peak_pos = peaks[0]
        
    return peak_pos