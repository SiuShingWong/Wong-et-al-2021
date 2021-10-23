import os
import numpy as np
import pandas as pd
from scipy.stats import describe
from centropy.analysis.dataframe import filter_merge_videos
from centropy.io import helper

def batch_descriptive_stat(input_dir, output_dir=None, attribute_dict=None, comparison_type='cycle', control=11, res_threshold=0.1, frame_rate=0.5, label_age=False):
    '''Obtain the description statistics for different parameters of different age types of different attributes
       Need a dictionary of for format {'attribute':['parameter1', 'parameters2', ...], ...}
    Args:
        input_dir: (str) where videos.csv and models.csv are saved
        output_dir: (str) directory to save statistics
        attribute_dict: {a0:[c0, c1, ...], a1:[...], ...}
            a: (str) 'intensity', 'area', 'density'
            c: (str) 'increase_rate_m', 'added_m', ...
        comparison_type: (str) 'cycle', 'manipulated_protein'
        control: (float) or (str) the control of comparison_type which we normalize to
        res_threshold: (float) threshold to remove outlier
        frame_rate: (float) conversion ratio from frame to time
        label_age: (bool) whether to calculate for old mothers and new mothers
    '''

    dataframe_dict = helper.read_data(input_dir, get_videos=True, get_models=True)
    dataframe_videos = dataframe_dict['videos']
    dataframe_models = dataframe_dict['models']
    if (dataframe_videos is None) or (dataframe_models is None): return "quit batch_descriptive_stat"
    if output_dir is None: output_dir = helper.construct_folders(input_dir)['Statistics']
    if attribute_dict is None: attribute_dict = { 'intensity': ['initial_r', 'peak_r', 'added_r','peak_time_r', 'increase_rate_r', 's-phase_duration'],}

    dataframe_videos = dataframe_videos[dataframe_videos['analyze']=='yes']
    merged_dataframe = filter_merge_videos(dataframe_videos, dataframe_models, res_threshold, frame_rate)
    df_descriptive = pd.DataFrame()
    for channel in merged_dataframe['channel'].unique():
        for attribute in list(attribute_dict.keys()):
            for parameter in attribute_dict[attribute]:
                df_all = descriptive_stat(merged_dataframe, attribute, parameter, channel, comparison_type, control, 'all')
                if label_age:
                    df_om = descriptive_stat(merged_dataframe, attribute, parameter, channel, comparison_type, control, 'old_mother')
                    df_nm = descriptive_stat(merged_dataframe, attribute, parameter, channel, comparison_type, control, 'new_mother')
                    df_descriptive = pd.concat((df_descriptive, df_all, df_om, df_nm))
                df_descriptive = pd.concat((df_descriptive, df_all))

    df_descriptive.dropna(subset=['mean'], inplace=True)
    df_descriptive.reset_index(inplace=True)

    df_descriptive.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'), index=False)


def descriptive_stat(merged_dataframe, attribute=None, parameter=None, channel=None, comparison_type=None, control=None, age_type=None):
    ''' Get descriptive statistics from these specific argument
    Args:
        merged_dataframe: (pandas.DataFrame) merged from models.csv and videos.csv
        attribute: (str) 'intensity', 'area', or 'density'
        parameter: (str) parameters e.g. 'added_m, 'increase_rate_m'
        channel: (int) channel
        comparison_type: (str) 'cycle', 'manipulated_protein'
        control: (float) or (str) the control of comparison_type which we normalize to
        age_type: (str) 'all', 'old_mother', 'new_mother'
    Return:
        df_descriptive: (pandas.DataFrame) summary statistics across sets
    '''
    df = merged_dataframe[(merged_dataframe['attribute']==attribute) & (merged_dataframe['age_type']==age_type) & (merged_dataframe['channel']==channel)]
    df_descriptive = pd.DataFrame()
    for experiment in df[comparison_type].unique():
        values = df[df[comparison_type]==experiment][parameter]
        stats = describe(values)
        temp =  pd.DataFrame(data={'attribute':attribute, 'channel':channel, 'parameter':parameter, comparison_type:experiment,
                                   'age_type':age_type, 'n':stats[0], 'min':stats[1][0], 'max':stats[1][1], 'mean':stats[2], 'variance':stats[3],
                                   'skewness':stats[4], 'kurtosis':stats[5]}, index=[0])
        df_descriptive = pd.concat((df_descriptive, temp))

    control_mean = df_descriptive[df_descriptive[comparison_type]==control]['mean'][0] # Get the mean value set we want to compare to
    df_descriptive['mean_norm_set'] = df_descriptive['mean']/control_mean # Normalize all mean by dividing to the mean of control set
    df_descriptive['mean_norm_age'] = np.nan
    
    return df_descriptive