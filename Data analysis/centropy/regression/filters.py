
import numpy as np 
# applies least-squares smoothing, can also do asymmetric baseline estimation. 
def baseline_als(y, border, sides, lam, p, niter=10):
    from scipy import sparse
    from scipy.sparse import linalg
    
    max_y = y.max()
    y_ = y/float(max_y)
    
    if border =='reflect' and sides=='single':
        y_ = np.hstack([y_[::-1], y_])
    if border =='reflect' and sides=='both':
        y_ =  np.hstack([y_[::-1], y_, y_[::-1]])
        
    L = len(y_)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y_)
        w = p * (y_ > z) + (1-p) * (y_ < z)
        
    if border =='reflect' and sides=='single':
        z = z[len(y):]
    if border =='reflect' and sides=='both':
        z = z[len(y):-len(y)]
        
    return max_y*z
    

def detrend_params(x,y):
    
    from scipy.stats import linregress
    
    params = linregress(x,y)
    ybase = params[0]*x + params[1]
    
    return y - ybase, ybase, params

    
def detrend_signal(time, signal, lam=1e6, p=0.1, niter=10, border=None, sides='both', method='linear'):
    
    baseline = baseline_als(signal, border=border, sides=sides, lam=lam, p=p, niter=niter)
    
    if method == 'linear':
        _,_, base_params = detrend_params(time, baseline)    
        signal_ =  signal - time*base_params[0]
    if method == 'nonlinear':
        signal_ = signal - baseline
        base_params = baseline
        
    return signal_, base_params    
    
    
def smooth_signal_GP(y, smoothing=0.75):
    
    import pymc3 as pm
    from theano import shared
    from pymc3.distributions.timeseries import GaussianRandomWalk
    
    LARGE_NUMBER = 1e5

    model = pm.Model()
    with model:
        smoothing_param = shared(0.9)
        mu = pm.Normal("mu", sd=LARGE_NUMBER)
        tau = pm.Exponential("tau", 1.0/LARGE_NUMBER)
        z = GaussianRandomWalk("z",
                               mu=mu,
                               tau=tau / (1.0 - smoothing_param),
                               shape=y.shape)
        obs = pm.Normal("obs",
                        mu=z,
                        tau=tau / smoothing_param,
                        observed=y)
        
    def infer_z(smoothing):
        with model:
            smoothing_param.set_value(smoothing)
            res = pm.find_MAP(vars=[z], method="L-BFGS-B")
            return res['z']
        
    # smoothing = 0.75
    y_smooth = infer_z(smoothing)
    
    return y_smooth
