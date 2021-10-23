import os
import numpy as np
import pandas as pd
from collections import Counter
from pingouin import multivariate_normality, pairwise_corr, corr
from centropy.statistics.verbrose import get_sig, get_corr
from centropy.analysis.dataframe import filter_merge_videos
from centropy.io import helper


def batch_correlation(input_dir, output_dir=None, pairwise_dict=None, comparison_type='cycle', res_threshold=0.1, frame_rate=0.5, label_age=False):
    ''' Perform correlation analysis of are columns provided by pairwise_dict
    Args:
        input_dir: (str) where videos.csv and models.csv are saved
        output_dir: (str) directory to save statistics
        pairwise_dict: (dict) {a0:[c0, c1, ...], a1:[...], ...}
            a: (str) e.g. 'area', 'intensity', or 'density'
            c: (str) parameters e.g. 'increase_rate_m', ...
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        res_threshold: (float) threshold to remove outlier
        frame_rate: (float) conversion ratio from frame to time
    '''
    dataframe_dict = helper.read_data(input_dir, get_videos=True, get_models=True)
    dataframe_videos = dataframe_dict['videos']
    dataframe_models = dataframe_dict['models']
    if (dataframe_videos is None) or (dataframe_models is None): return "quit batch_descriptive_stat"
    if output_dir is None: output_dir = helper.construct_folders(input_dir)['Statistics']
    if pairwise_dict is None: pairwise_dict = { 'intensity': ['initial_r', 'peak_r', 'added_r','peak_time_r', 'increase_rate_r', 's-phase_duration'],}

    if label_age: age_list = ['all','old_mother','new_mother']
    else: age_list = ['all']

    merged_dataframe = filter_merge_videos(dataframe_videos, dataframe_models, res_threshold, frame_rate)

    df_corr = pd.DataFrame()
    for channel in merged_dataframe['channel'].unique():
        for attribute in list(pairwise_dict.keys()): # Loop through attribute
            for age_type in age_list: # Loop through age type
                    df_temp = pairwise_columns_correlation(merged_dataframe, channel, age_type, attribute, comparison_type, pairwise_dict)
                    df_corr = pd.concat((df_corr, df_temp))

    df_corr.reset_index(inplace=True)
    df_corr.drop(labels='index', axis=1, inplace=True)

    df_corr.to_csv(os.path.join(output_dir, 'correlations.csv'), index=False)


def pairwise_columns_correlation(merged_dataframe, channel=0, age_type='all', attribute='intensity', comparison_type='cycle', pairwise_dict=None):
    ''' Perform correlation analysis between any pair of parameters in different experiment set
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        channel: (int) channel
        age_type: (str) 'all', 'old_mother', 'new_mother'
        attribute: (str) 'area', 'intensity', ...
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        pairwise_dict: (dict) {a0:[c0, c1, ...], a1:[...], ...}
            a: (str) e.g. 'area', 'intensity', or 'density'
            c: (str) parameters e.g. 'increase_rate_m', ...
    Return:
        (pandas.DataFrame) updated df_corr
    '''
    cond_attribute = merged_dataframe['attribute']==attribute
    cond_age = merged_dataframe['age_type']==age_type
    cond_channel = merged_dataframe['channel']==channel

    cache = []
    df_corr = pd.DataFrame()
    # Loop through 2 pairs of parameters
    for col_1 in pairwise_dict[attribute]:
        for col_2 in pairwise_dict[attribute]:
            if set((col_1, col_2)) in cache: continue # Don't calculate again
            if col_1 == col_2: continue # Don't calculate when the pair is from the same data

            for individual_set in merged_dataframe[comparison_type].unique(): # loop through individual set
                df = merged_dataframe[cond_attribute & cond_age & cond_channel & (merged_dataframe[comparison_type]==individual_set)]
                temp = pairwise_correlation(df, channel, age_type, attribute, comparison_type, individual_set, col_1, col_2)
                df_corr = df_corr.append(temp, ignore_index=True)

            df = merged_dataframe[cond_attribute & cond_age & cond_channel] # Get the data from a particular attribute, age_type for the whole set
            temp = pairwise_correlation(df, channel, age_type, attribute, comparison_type, np.nan, col_1, col_2)
            df_corr = df_corr.append(temp, ignore_index=True)

            cache.append(set((col_1, col_2))) # Indicate we have visisted the pair

    # To calculate the Berforroni correction factors for multiple comparison
    cache = [tuple(e) for e in cache]
    bonf_factor = len(Counter(cache).keys()) # Get the total number of comparison
    
    # Appending additional information
    df_corr['correction_method'] = 'bonf'
    df_corr['correction_factor'] = bonf_factor
    df_corr['p-val'] = df_corr['p-unc']*bonf_factor
    df_corr['correlation'] = df_corr['r'].apply(get_corr)
    df_corr['significance'] = df_corr['p-val'].apply(get_sig)

    return df_corr


def pairwise_correlation(merged_dataframe, channel=None, age_type=None, attribute=None, comparison_type=None, individual_set=None, col_1=None, col_2=None):
    ''' Perform correlation statistics on 2 columns
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        df_corr: (pandas.DataFrame) store correlation statistics
        channel: (int) channel
        age_type: (str) 'all', 'old_mother', 'new_mother'
        attribute: (str) 'area', 'intensity', ...
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        individual_set: (str) the set to perform correlation analysis
        col_1, col_2: (str) parameters to analyze
    Return:
        (pandas.DataFrame)
    '''
    is_normal, _ = multivariate_normality(merged_dataframe[[col_1, col_2]]) # If the bivariate data are noramlly distributed

    if is_normal: corr_test = pairwise_corr(merged_dataframe, columns=[col_1, col_2], covar=None, tail='two-sided', method='pearson', padjust='none', nan_policy='pairwise')
    else: corr_test = pairwise_corr(merged_dataframe, columns=[col_1, col_2], covar=None, tail='two-sided', method='spearman', padjust='none', nan_policy='pairwise')

    # print (corr_test['r'].values[0])

    return pd.DataFrame({'channel':channel, 'age_type':age_type, 'attribute':attribute, 'A': col_1, 'B': col_2, comparison_type: individual_set, 
    'normality':'Henze-Zirkler', 'is_normal':is_normal, 'test':corr_test['method'].values[0], 'sample_size':corr_test['n'].values[0], 
    'r':corr_test['r'].values[0], 'p-unc':corr_test['p-unc'].values[0]}, index=[0])
