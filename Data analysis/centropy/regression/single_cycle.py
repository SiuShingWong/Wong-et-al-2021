import os
from tqdm import tqdm

import pandas as pd
import numpy as np

from centropy.regression import signal
from centropy.io import helper

# =====================================================================================================
# ================================== Batch API for regression =========================================
# =====================================================================================================


def batch_fitting(input_dir, output_dir=None, frame_rate=0.5, peak_trough_prominence=0.5, smoothing=.6):
    ''' Batch model regression for single cycle video ** Special notes on the naming convention in the videos.csv in README.md
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        output_dir: (str) directory containing videos.csv and centrosomes.csv
        time_factor: (float) conversion ratio from frame to time
        peak_trough_prominence: (float) the minimum difference between peak and through to be regarded as a peak
        smoothing: (float) smoothing of curve
    '''
    ################ Initialization of dataframe ################
    if not output_dir: output_dir = input_dir

    dataframe_dict = helper.read_data(input_dir, get_videos=True, get_centrosomes=True)
    dataframe_videos = dataframe_dict['videos']
    dataframe_centrosomes = dataframe_dict['centrosomes']
    if (dataframe_videos is None) or (dataframe_centrosomes is None): return "quite batch_categorical"

    dataframe_models = initialize_models()
    dataframe_simulations = initialize_simulations()
    #############################################################
    
    ############ Main loop of regression for each video ############
    for attribute in ['intensity', 'area', 'density']: # major attribute: can be extended in the future
        
        adjust_model = False
        
        if 'start_end' in dataframe_videos.columns: adjust_model = True # indicate whether to trim the region for fitting
        if not 'model_'+attribute+'_all' in dataframe_videos.columns: continue
            
        for _, row in tqdm(dataframe_videos.iterrows()): # loop over videos
            video_name = row['video_name']
            channel = int(row['channel'])
            cs_frame = int(row['man_cs'])
            
            for suffix in ['_all', '_om', '_nm']: # for every centrosomes, old mothers, and new mothers
                model_key = 'model_'+attribute+suffix
                if model_key in dataframe_videos.columns:

                    p0_key = 'init_'+attribute+suffix
                    if p0_key in dataframe_videos.columns: p0 = get_p0(p0_str=row[p0_key]) # initial guess of the fitting parameters
                    else: p0 = None
                    
                    x_raw, y_raw, age_type = extract_fitting_dataframe(dataframe_centrosomes, video_name, channel, cs_frame, frame_rate, attribute, suffix)
                    
                    if adjust_model: x, y, start_end = adjust_start_end(x_raw, y_raw, row['start_end']) # trim the region for fitting
                    else: x, y = x_raw, y_raw
                    
                    if not y.any(): continue
                    initial_r, peak_r, added_r, peak_time_r, increase_rate_r, decrease_rate_r = manual_parameters(x, y, frame_rate) # manual features
                    
                    if (row[model_key] in ['linear', 'linear_plateau', 'linear_piecewise', 'single_oscillation']):
                        initial_m, peak_m, peak_time_m, increase_rate_m, decrease_rate_m, added_m, max_rate_m, max_rate_time_m, residual = [np.nan for i in range(9)]
                        
                        if row[model_key] == 'linear': # linear features
                            xp, yp, arg_x = fine_sampling(x, y, 0.1)
                            model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual = linear_parameters(xp, yp, p0)
                        elif row[model_key] == 'linear_plateau': # linear plateau features
                            xp, yp, arg_x = fine_sampling(x, y, 0.1)
                            model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual = linear_plateau_parameters(xp, yp, p0)
                        elif row[model_key] == 'linear_piecewise': # linear piecewise features
                            xp, yp, arg_x = fine_sampling(x, y, 0.1)
                            model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, decrease_rate_m, residual = linear_piecewise_parameters(xp, yp, p0)
                        elif row[model_key] == 'single_oscillation': # mixed gaussian features
                            xp, yp, arg_x = fine_sampling(x, y, 0.025)
                            model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, max_rate_m, max_rate_time_m, residual = single_oscillation_parameters(x=xp, y=yp, tick=0.25, peak_trough_prominence=peak_trough_prominence, smoothing=smoothing)
                            
                        dataframe_models = dataframe_models.append({
                            'video_name':video_name, 'attribute':attribute, 'age_type':age_type, 'channel': channel, 'model_type':row[model_key], 
                            'initial_r':initial_r, 'peak_r':peak_r, 'added_r': added_r,'peak_time_r':peak_time_r, 'increase_rate_r':increase_rate_r, 'decrease_rate_r': decrease_rate_r, 
                            'initial_m':initial_m, 'peak_m':peak_m, 'added_m': added_m, 'peak_time_m':peak_time_m, 'increase_rate_m':increase_rate_m, 'decrease_rate_m':decrease_rate_m,
                            'max_rate_m':max_rate_m, 'max_rate_time_m':max_rate_time_m, 'residual':residual}, ignore_index=True)
                        model = model[arg_x]
                        if adjust_model:
                            prepend, append = np.array([np.nan, ]*abs(start_end[0])), np.array([np.nan, ]*abs(start_end[1]))
                            model = np.concatenate((prepend, model, append))
                        dataframe_simulations = log_simulation(dataframe_simulations, video_name, channel, age_type, attribute, x_raw, model)
                    else:
                        dataframe_models = dataframe_models.append({
                            'video_name':video_name, 'attribute':attribute, 'age_type':age_type, 'channel': channel, 
                            'initial_r':initial_r, 'peak_r':peak_r, 'added_r': added_r,'peak_time_r':peak_time_r, 'increase_rate_r':increase_rate_r, 'decrease_rate_r': decrease_rate_r, 'residual':0}, ignore_index=True)
                        dataframe_simulations = initialize_simulations()
    #############################################################
        
    ############ Save models.csv and simulations.csv ############
    dataframe_simulations['area_norm'] = dataframe_simulations['area_um2'] / dataframe_simulations['area_um2'].min()
    dataframe_simulations['mean_intensity'] = dataframe_simulations['total_intensity_norm'] / dataframe_simulations['area_um2']
    dataframe_simulations['mean_intensity_norm'] = dataframe_simulations['total_intensity_norm'] / dataframe_simulations['area_norm']
    dataframe_simulations.to_csv(os.path.join(output_dir, 'simulations.csv'), index=False)
    dataframe_models = dataframe_models.sort_values(by=['attribute', 'age_type', 'channel', 'video_name'])
    dataframe_models.to_csv(os.path.join(output_dir, 'models.csv'), index=False)
    #############################################################


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================


# =====================================================================================================
# ================================= Parameters from regression ========================================
# =====================================================================================================


def manual_parameters(x, y, frame_rate=0.5):
    ''' Extract parameters from manual spotting of data
    Args:
        y: (list) or (1D numpy.array) y values
        frame_rate: (float) conversion ratio from frame to time
    Return:
        initial_r, peak_r, added_r, peak_time_r, increase_rate_r: (float)
    '''
    peak_r, final_r = y.max(), y[-1]
    peak_time_r = np.argmax(y) * frame_rate
    increase_rate_r = (peak_r - y[0]) / peak_time_r
    
    initial_r = y[0] - increase_rate_r * x[0]
    added_r = peak_r - initial_r
    
    if (len(y) - peak_time_r) < 1: decrease_rate_r = 0
    else: decrease_rate_r = abs((final_r - peak_r) / (len(y) - peak_time_r))

    return initial_r, peak_r, added_r, peak_time_r, increase_rate_r, decrease_rate_r


def linear_parameters(x, y, p0=None, fit_loss='huber'):
    ''' Extract parameters from linear fitting
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        p0: (list) of (float) initial guess of parameters, with length of 2
        fit_loss: (str) cost function
    Return:
        initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual: (float)
    '''
    if p0 is None: p0 = [1, 0]# If no p0 is provided
    fitter = signal.LinearSignal(x=x, y=y)
    fitter.fit(p0_linear=p0, fit_loss=fit_loss)
    model = fitter.regress_y
    # Parameters
    peak_m = model[-1]
    peak_time_m = x[-1]
    increase_rate_m, _ = fitter.regress_params
    initial_m = model[0] - increase_rate_m*x[0]
    added_m = np.nan
    if (peak_m != np.nan) and (initial_m != np.nan): added_m = peak_m - initial_m
    residual = fitter.cost

    return model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual


def linear_plateau_parameters(x, y, p0=None, fit_loss='huber'):
    ''' Extract parameters from linear plateau fitting
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        p0: (list) of (float) initial guess of parameters, with length of 3
        fit_loss: (str) cost function
    Return:
        initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual: (float)
    '''
    if p0 is None: p0 = [0.5, 0.8, 1]
    p0[0] = p0[0]*x.max() # Adjust the guess of the peak
    fitter = signal.LinearPlateauSignal(x=x, y=y)
    fitter.fit(p0_piecewise=p0, fit_loss=fit_loss)
    model = fitter.regress_y
    # Parameters
    peak_time_m, peak_m, increase_rate_m = fitter.regress_params
    initial_m = model[0] - increase_rate_m*x[0]
    added_m = np.nan
    if (peak_m != np.nan) and (initial_m != np.nan): added_m = peak_m - initial_m
    residual = fitter.cost
    
    return model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, residual


def linear_piecewise_parameters(x, y, p0=None, fit_loss='huber'):
    ''' Extract parameters from linear piecewise fitting
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        p0: (list) of (float) initial guess of parameters, with length of 4
        fit_loss: (str) cost function
    Return:
        initial_m, peak_m, added_m, peak_time_m, increase_rate_m, decrease_rate_m, residual: (float)
    '''
    if p0 is None: p0 = [0.8, 0.8, 1, -1]
    p0[0] = p0[0]*x.max() # Adjust the guess of the peak
    # Fitting linear piecewise
    # Todo: Introduce constraint of slopes
    fitter = signal.PiecewiseLinearSignal(x=x, y=y)
    fitter.fit(p0_piecewise=p0, fit_loss=fit_loss)
    model = fitter.regress_y
    # Parameters
    peak_time_m, peak_m, increase_rate_m, decrease_rate_m = fitter.regress_params
    initial_m = model[0] - increase_rate_m*x[0]
    added_m = np.nan
    if (peak_m != np.nan) and (initial_m != np.nan): added_m = peak_m - initial_m
    residual = fitter.cost

    return model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, decrease_rate_m, residual


def single_oscillation_parameters(x, y, smoothing=.6, peak_trough_prominence=0.5, fit_loss='cauchy', tick=0.1):
    ''' Extract parameters from linear piecewise fitting
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        smoothing: (float) smoothing of curve
        peak_trough_prominence: (float) the minimum difference between peak and through to be regarded as a peak
        fit_loss: (str) cost function
    Return:
        initial_m, peak_m, added_m, peak_time_m, increase_rate_m, max_rate_m, max_rate_time_m, residual: (float)
    '''
    fitter = signal.CycleSignal(x=x, y=y, single_cycle=True)
    fitter.fit_multi_gauss(smoothing=smoothing, peak_trough_prominence=peak_trough_prominence, fit_loss=fit_loss)
    fitter.extract_peak_increase()
    fitter.extract_peak_decrease()
    model = fitter.gauss_fit_signal
    # Parameters
    initial_m = model[0]
    peak_time_m = 0
    peak_m = 0
    for temp in fitter.peak_index:
        if model[temp] > peak_m:
            peak_m = model[temp]
            peak_time_m = temp
    peak_time_m = peak_time_m * tick # It's the size of the sampling but need to adjust for input
    increase_rate_m = (peak_m - initial_m) / peak_time_m
    if fitter.peak_increase[0].any():
        max_rate_m = 0
        max_rate_time_m = 0
        for i in range(len(fitter.peak_increase[0])):
            if fitter.peak_increase[1][i] > max_rate_m:
                max_rate_m = fitter.peak_increase[1][i]
                max_rate_time_m = fitter.peak_increase[0][i]
    max_rate_m = max_rate_m / tick
    max_rate_time_m = max_rate_time_m * tick # It's the size of the sampling but need to adjust for input
    residual = fitter.cost
    added_m = np.nan
    if (peak_m != np.nan) and (initial_m != np.nan):
        added_m = peak_m - initial_m

    return model, initial_m, peak_m, added_m, peak_time_m, increase_rate_m, max_rate_m, max_rate_time_m, residual


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================


# =====================================================================================================
# ================================== Helper function for API ==========================================
# =====================================================================================================


def get_p0(p0_str='_'):
    ''' Obtain p0 from p0_str in the videos.csv
    Args:
        p0_str: (str) should be either '_', 'n_n', 'n_n_n', 'n_n_n_n', where n is float and dependent on the models to be fitted
    Return:
        1) None or 2)(list) of (float) of p0 for regression
    '''
    if len(p0_str)==1: # Return None if encounter '_' --> default p0 for corresponding model
        return None
    else:
        return list(map(float, p0_str.split('_')))


def extract_fitting_dataframe(dataframe_centrosomes, video_name, channel, cs_frame, frame_rate, attribute, suffix):
    ''' Obtain data for fitting
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        video_name: (str) video name of this video
        channel: (int) current channel
        attribute: (str) should be 'area', 'intensity', 'distance', or 'mean_intensity'
        suffix: (str) should be '_all', '_om', _nm
    Return:
        x_raw: (list) or (1D numpy.array) x values
        y_raw: (list) or (1D numpy.array) y values
    '''
    attribute_dict = {'intensity': 'total_intensity_norm', 'area': 'area_um2', 'density': 'mean_intensity_norm', 'distance':'distance_um'}
    cond_video = dataframe_centrosomes['video_name']==video_name
    cond_channel = dataframe_centrosomes['channel']==channel
    cond_frame = dataframe_centrosomes['frame'] >= cs_frame - 1
    
    if suffix == '_all':
        df = dataframe_centrosomes[cond_video & cond_channel & cond_frame]
        age_type = 'all'
    elif suffix =='_om':
        df = dataframe_centrosomes[cond_video & cond_channel & cond_frame & (dataframe_centrosomes['age_type']=='old_mother')]
        age_type = 'old_mother'
    elif suffix == '_nm':
        df = dataframe_centrosomes[cond_video & cond_channel & cond_frame & (dataframe_centrosomes['age_type']=='new_mother')]
        age_type = 'new_mother'
        
    # df['time'] = df['time'] - cs_frame * frame_rate
    attribute_value = df.pivot(index='time', columns='particle', values=attribute_dict[attribute])
    
    x_raw = attribute_value.index.values
    y_raw = attribute_value.mean(axis=1, skipna=True).values

    return x_raw, y_raw, age_type


def adjust_start_end(x, y, start_end_str='_'):
    ''' Obtain x and y of respective region where regression is performed
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        start_end_str: (str) should be either '_', '_n', 'n_', 'n_n', where n is float indicating the start or end
    Return:
        x: (list) or (1D numpy.array) updated x values
        y: (list) or (1D numpy.array) updated y values
        [start, end]: (list) of start position and end position
    '''
    if len(start_end_str)==1: # '_' use all the data for fitting
        return x, y, [0, 0]
    elif start_end_str.startswith('_'):
        start, end = 0, int(start_end_str.split('_')[-1]) # '_n' use data until n
        x, y = x[0: end], y[0: end]
    elif start_end_str.endswith('_'): # 'n_' use data starting from n
        start, end = int(start_end_str.split('_')[0]), 0
        x, y = x[start:], y[start:]
    else: # 'n_n' use data within n and n
        start, end = list(map(int, start_end_str.split('_')))
        x, y = x[start:end], y[start:end]
    return x, y, [start, end]


def fine_sampling(x, y, tick=0.1):
    ''' Change the x sampling, and interpolate on y
    Args:
        x: (list) or (1D numpy.array) x values
        y: (list) or (1D numpy.array) y values
        tick: (float) the sampling ratio
    Returns:
        xp: (list) or (1D numpy.array) finer x values
        yp: (list) or (1D numpy.array) finer y values
        arg_x: (list) or (1D numpy.array) the position of original x-values in the new finer x-values
    '''
    xp = np.arange(x[0], x[-1] + tick, tick) # Increase sampling by TICK_LENGTH
    xp = np.round(xp, 1) # round the value to nearest decimal
    yp = np.interp(xp, x, y) # interpolate y using finer x
    # Find out the position of X in the fine X
    arg_x = []
    for e in x:
        arg_x.append(np.where(xp==e)[0][0])
    return xp, yp, arg_x


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================


# =====================================================================================================
# ================================== Initialization of dataframe ======================================
# =====================================================================================================


def initialize_models():
    ''' Initialize models.csv to store parameters of regression (r denote manual parameters, m denotes parameters from modelling)
    '''
    return pd.DataFrame(columns=['video_name', 'channel', 'attribute', 'age_type', 'particle_type', 'model_type',
                                 'initial_r', 'peak_r', 'added_r', 'peak_time_r', 'increase_rate_r', 'decrease_rate_r', 
                                 'initial_m', 'peak_m', 'added_m', 'peak_time_m', 'increase_rate_m', 'decrease_rate_m',
                                 'max_rate_m', 'max_rate_time_m', 'residual'])


def initialize_simulations():
    ''' Initializ simulations.csv to store the simulated centrosomes under the the model
    '''
    return pd.DataFrame(columns=['video_name', 'channel', 'particle_type', 'age_type', 'mean_intensity', 'mean_intensity_norm', 
                                 'distance', 'distance_um', 'area', 'area_um2', 'total_intensity', 'total_intensity_norm', 'time'])


def log_simulation(dataframe_simulations, video_name=None, channel=0, age_type=None, attribute=None, x_raw=None, model=None):
    ''' Add simulation data into simulation.csv. It has a similar column name as centrosoms.csv (#TODO reduce the column)
    Args:
        dataframe_simulations: (pandas.DataFrame) simulations.csv
        video_name: (str) video name
        channel: (int) the current channel
        age_type: (str) 'old_mother' or 'new_mother'
        attribute: (str) attribute selection e.g. 'area', 'total_intensity'
        x_raw: (list) or (numpy.array) time of the simulation
        model: (list) or (numpy.array) simulated features dynamics
    Return:
        updated simulations.csv
    '''
    attribut_dict = {'intensity':'total_intensity_norm', 'area':'area_um2', 'density':'mean_intensity_norm', 'distance':'distance_um'}
    column_name = attribut_dict[attribute]
    cond = (dataframe_simulations['video_name']==video_name)&(dataframe_simulations['particle_type']=='simulation')&(dataframe_simulations['age_type']==age_type)&(dataframe_simulations['channel']==channel)
    if cond.any():
        dataframe_simulations.loc[cond, column_name] = model
    else:
        temp_df = initialize_simulations()
        temp_df['time'] = x_raw
        temp_df[column_name] = model
        temp_df['video_name'] = video_name
        temp_df['channel'] = channel
        temp_df['age_type'] = age_type
        temp_df['particle_type'] = 'simulation' 
        dataframe_simulations = dataframe_simulations.append(temp_df)
    return dataframe_simulations


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================