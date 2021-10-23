def get_sig(p_val=None):
    '''Stratify p value into asterisk'''
    import numpy as np
    if p_val is None: return 'no p_val'

    if p_val <= 0.0001: return '****'
    elif p_val <= 0.001: return '***'
    elif p_val <= 0.01: return '**'
    elif p_val <= 0.05: return '*'
    elif p_val > 0.05: return 'ns'
    else: return np.nan

def get_corr(r=None):
    '''Stratify r value into expression'''
    if r is None: return 'no r_val'
    
    if r == 0: return 'no corr'
    elif r < 0: subfix = 'negative'
    else: subfix = 'positive'
    
    if abs(r) >= 0.66: prefix = 'strong'
    elif abs(r) >= 0.33: prefix = 'moderate'
    elif abs(r) > 0: prefix = 'weak'
    
    return '{}_{}'.format(prefix, subfix)