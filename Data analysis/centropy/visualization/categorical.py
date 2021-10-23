import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from centropy.io.helper import construct_folders, read_data
from centropy.analysis.dataframe import filter_merge_videos


def batch_box(input_dir, output_dir, comparison_type, attribute_dict, between_type, frame_rate, residue_threshold, bottom=0):
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
        bottom: (float) the minimum value of y-axis
    '''
    dataframe_dict = read_data(input_dir, get_videos=True, get_models=True)
    dataframe_videos, dataframe_models = dataframe_dict['videos'], dataframe_dict['models']
    if (dataframe_videos is None) or (dataframe_models is None): return 'batch_scatter'
    if output_dir is None: output_dir = construct_folders(input_dir)['Categorical']
    if attribute_dict is None: attribute_dict = { 'intensity': ['initial_r', 'peak_r', 'added_r','peak_time_r', 'increase_rate_r', 's-phase_duration'],}

    merged_dataframe = filter_merge_videos(dataframe_videos, dataframe_models, residue_threshold, frame_rate)
    for attribute in list(attribute_dict.keys()):
        save_dir = os.path.join(output_dir, attribute)
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        for channel in dataframe_videos['channel'].unique():
            for parameter in attribute_dict[attribute]:
                box(merged_dataframe, save_dir, comparison_type, channel, attribute, parameter, 
                    between_type, frame_rate, bottom)

    pass


def box(merged_dataframe, output_dir=None, comparison_type=None, channel=None, attribute=None, parameter=None, 
            between_type=None, frame_rate=0.5, bottom=0, top=10):
    ''' Generate a box plot for a parameter of an attribute
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        comparison_type: (str) can be 'cycle' or 'manipulated_protein'
        channel: (int) 0 or 1 indicating the channel
        attribute: (str) e.g. 'intensity', 'area'
        parameter: (str) e.g. 'initial_r' see the column name of models.csv
        between_type: (str) 'age_type'
        frame_rate: (float)
        residue_threshold: (float) the quality of the fitting
        bottom: (float) the minimum value of y-axis
    '''
    df = select_data(merged_dataframe, comparison_type, attribute, channel, between_type)
    xlabel, ylabel, order, bar_color, hue_color = cat_plot_styling(df, comparison_type)

    fig = plt.figure(figsize=(5,8), dpi=100)
    plt.style.use('ggplot')
    if between_type is None: 
        ax = sns.boxplot(x=comparison_type, y=parameter, data=df, order=order, fliersize=0)
        ax = sns.swarmplot(x=comparison_type, y=parameter, data=df, order=order, color='black')
    else: 
        ax = sns.boxplot(x=comparison_type, y=parameter, hue=between_type, data=df, order=order, fliersize=0)
        ax = sns.swarmplot(x=comparison_type, y=parameter, hue=between_type, data=df, order=order, color='black', dodge=True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])

    for _, box in enumerate(ax.artists):
        box.set_edgecolor('k')

    plt.ylabel(ylabel[attribute][parameter], family='Arial', fontsize='large')
    plt.xlabel(xlabel[comparison_type], family='Arial', fontsize='large')
    plt.yticks(family='Arial', fontsize='large')
    plt.xticks(family='Arial', fontsize='large')
    plt.ylim(bottom=bottom)
    plt.tight_layout()
    plt.margins(x=0)

    save_name = os.path.join(output_dir, f'Channel-{channel}_{between_type}_{comparison_type}_{attribute}_{parameter}.png')
    fig.savefig(save_name, facecolor='white')

    plt.ioff()
    plt.close('all')


def select_data(merged_dataframe, comparison_type=None, attribute=None, channel=0, between_type=None):
    df = merged_dataframe[merged_dataframe['attribute']==attribute]
    if comparison_type=='cycle' or comparison_type=='manipulated_protein':
        if between_type is None:
            df = df[df['channel']==channel]
            df['channel'].cat.remove_unused_categories(inplace=True)
            df = df[df['age_type'].isin(['all'])]
            df['age_type'].cat.remove_unused_categories(inplace=True)
        elif between_type == 'age_type': 
            df = df[df['channel']==channel]
            df['channel'].cat.remove_unused_categories(inplace=True)
            df = df[df['age_type'].isin(['old_mother', 'new_mother'])]
            df['age_type'].cat.remove_unused_categories(inplace=True)
        elif between_type == 'channel':
            df = df[df['age_type'].isin(['all'])]
            df['age_type'].cat.remove_unused_categories(inplace=True)
    return df


def cat_plot_styling(merged_dataframe, comparison_type=None):
    ''' A collection of dictionary of categorical plot parameters
    Args:
        merged_dataframe: (pandas.DataFrame) a merged dataframe from videos.csv and models.csv
        comparison_type: (str) 'cycle' or 'manipulated_protein'
    '''
    # Dictionary mapping attribute and parameters into their corresponding y axis title
    ylab_dict = {
        'area':{'initial_r':'Manual Initial Area(um2)', 'peak_r':'Manual Peak Area(um2)',
                'added_r':'Manual Area Added(um2)', 'peak_time_r':'Manual Area Growth Period(min)',
                'increase_rate_r':'Manual Area Growth Rate(um2/min)', 'decrease_rate_r':'Manual Area Decrease Rate(um2/min)',
                'initial_m':'Modelled Initial Area(um2)',
                'peak_m':'Modelled Peak Area(um2)', 'added_m':'Modelled Area Added(um2)', 
                'peak_time_m':'Modelled Area Growth Period(min)', 'increase_rate_m':'Modelled Area Growth Rate(um2/min)',
                'decrease_rate_m':'Modelled Area Compaction Rate(um2/min)', 'max_rate_m':'Modelled Maximum Area Growth Rate(um2/min)',
                'max_rate_time_m':'Modelled Maximum Area Growth Period(min)', 'residual':'Area Fitting Quality(AU)', 's-phase_duration':'S-phase Length(min)',},
        'intensity':{'initial_r':'Manual Initial Total Intensity(AU)', 'peak_r':'Manual Peak Total Intensity(AU)',
                    'added_r':'Manual Total Intensity Added(AU)', 'peak_time_r':'Manual Total Intensity Growth Period(min)',
                    'increase_rate_r':'Manual Total Intensity Growth Rate(AU)', 'decrease_rate_r':'Manual Total Intensity Decrease Rate(AU)',
                    'initial_m': 'Modelled Initial Total Intensity(AU)',
                    'peak_m':'Modelled Peak Total Intensity(AU)', 'added_m':'Modelled Total Intensity Added(AU)',
                    'peak_time_m':'Modelled Total Intensity Growth Period(min)', 'increase_rate_m':'Modelled Total Intensity Growth Rate(AU)',
                    'decrease_rate_m':'Modelled Total Intensity Decline Rate(AU)', 'max_rate_m':'Modelled Maximum Total Intensity Growth Rate(AU)',
                    'max_rate_time_m':'Modelled Maximum Total Intensity Growth Period(AU)', 'residual':'Total Intensity Fitting Quality(AU)',
                    's-phase_duration':'S-phase Length(min)'},
        'density':{},}
    # Dictionary mapping comparison_type into their corresponding x axis title
    xlab_dict = {'cycle':'Cycle', 'manipulated_protein':'Genotype'}
    # Dictionary mapping comparison_type into their corresponding color palettes
    pallete_dict = {'cycle':sns.light_palette("yellow", reverse=True), 'manipulated_protein':sns.color_palette("muted")}
    # Dictionary mapping age_type or color channel into their color palettes
    hue_color_dict = {0:'#191970', 1:'#8b0000', 'old_mother':'#1e90ff',  'new_mother':'#ff1493'}
    # The order of arrangement of categorical plots
    comparison_sets = merged_dataframe[comparison_type].unique()
    if comparison_type == 'manipulated_protein':
        order = ['Control',]
        for e in comparison_sets:
            if e != 'Control':
                order.append(e)
    else:
        order = list(comparison_sets)
    return xlab_dict, ylab_dict, order, pallete_dict, hue_color_dict