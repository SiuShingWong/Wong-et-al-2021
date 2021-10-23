import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from centropy.io.helper import construct_folders, read_data
from centropy.analysis.dataframe import filter_merge_videos
from centropy.visualization.categorical import select_data

def pairwise_scatter(input_dir, output_dir, comparison_type, attribute_dict, between_type, frame_rate, residue_threshold):
    ''' Generate box plots for different parameters of an attribute
    Args:
        input_dir: (str) the path that contains videos.csv and centrosomes.csv
        output_dir: (str) the directory where you want to save the figures. If None is present, it will point to the input_dir
        comparison_type: (str) can be 'cycle' or 'manipulated_protein'
        attribute_dict: (dict) {a0:[c0, c1, ...], a1:[...], ...}
            a: (str) e.g. 'area', 'intensity', or 'density'
            c: (str) parameters e.g. 'increase_rate_m', ...
        between_type: (str) 'age_type'
        frame_rate: (float)
        residue_threshold: (float) the quality of the fitting
    '''
    dataframe_dict = read_data(input_dir, get_videos=True, get_models=True)
    dataframe_videos, dataframe_models = dataframe_dict['videos'], dataframe_dict['models']
    if (dataframe_videos is None) or (dataframe_models is None): return 'batch_scatter'
    if output_dir is None: output_dir = construct_folders(input_dir)['Correlation']
    if attribute_dict is None: attribute_dict = { 'intensity': ['initial_r', 'peak_r', 'added_r','peak_time_r', 'increase_rate_r', 's-phase_duration'],}

    merged_dataframe = filter_merge_videos(dataframe_videos, dataframe_models, residue_threshold, frame_rate)

    for channel in dataframe_videos['channel'].unique():
        for attribute in list(attribute_dict.keys()):
            df = select_data(merged_dataframe, comparison_type, attribute, channel, between_type)
            plt.style.use('ggplot')
            if between_type is None:
                g = sns.pairplot(data=df, vars=attribute_dict[attribute], dropna=True, hue=comparison_type, palette="husl", diag_kws = {'alpha':0.55, 'bins':20},  diag_kind='hist', corner=True)
            else:
                g = sns.pairplot(data=df, vars=attribute_dict[attribute], dropna=True, hue=between_type, palette="husl", diag_kws = {'alpha':0.55, 'bins':20}, diag_kind='hist', corner=True)
        if between_type is None: save_name = os.path.join(output_dir, f'Channel-{channel}_{attribute}.png')
        else: save_name = os.path.join(output_dir, f'Channel-{channel}_{attribute}_between-{between_type}.png')
        g.savefig(save_name, facecolor='white')
        plt.ioff()
        plt.close('all')
