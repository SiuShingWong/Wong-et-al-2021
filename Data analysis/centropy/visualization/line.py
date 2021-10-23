import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from centropy.io.helper import construct_folders, read_data
from centropy.analysis.dataframe import mean_sd_centrosomes, data_selection, filter_merge_videos_prism


def all_embryos(input_dir, output_dir, comparison_type, attribute, label_age, alignment):
    ''' Obtain a overlayed mean of all embryos in a dataset from the same condition
    Args:
        input_dir: (str) the path that contains videos.csv and centrosomes.csv
        output_dir: (str) the directory where you want to save the figures. If None is present, it will point to the input_dir
        comparison_type: (str) can be 'cycle' or 'manipulated_protein'
        attribute: (str) e.g. 'total_intensity_norm' from centrosomes.csv column
        label_age: (bool) whether to show old_mother and new_mother
        alignment: (str) 'cs' aligns to centrosome separation, 'neb' aligns to NEB
    '''
    # Read required data from input_dir
    dataframe_dict = read_data(input_dir, get_videos=True, get_centrosomes=True)
    dataframe_videos, dataframe_centrosomes = dataframe_dict['videos'], dataframe_dict['centrosomes']
    if (dataframe_videos is None) or (dataframe_centrosomes is None): return "quit all_embryos"

    # Generate output_dir if users didn't provide it
    if output_dir is None: output_dir = construct_folders(input_dir)['Collective']
    save_dir = os.path.join(output_dir, attribute)
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    # age_types based on whether to show different age type of centrosomes
    if label_age: age_types = ['new_mother', 'old_mother']
    else: age_types = ['all']

    experiments = dataframe_videos[comparison_type].unique() 
    merged_dataframe = filter_merge_videos_prism(dataframe_videos, dataframe_centrosomes, res_threshold=1, frame_rate=0.5)
    num_channel = dataframe_videos['num_channel'].unique()[0]
    line_colors_dict, cycle_dim_dict, ylab_dict = line_plot_styling()

    for experiment in experiments: # Loop through different experimental condition provided in videos.csv

        df = dataframe_videos[dataframe_videos[comparison_type]==experiment]
        neb = df['s-phase_duration'].mean()
        cycle = df['cycle'].unique()[0]

        plt.style.use('ggplot')
        fig, ax0 = plt.subplots(figsize=(cycle_dim_dict[int(cycle)][1],5), dpi=150)

        if num_channel==1: axs = [ax0]
        elif num_channel==2: axs = [ax0, ax0.twinx()]

        for channel in dataframe_videos['channel'].unique():
            
            min_frame = 10000
            max_frame = 0
            max_y = 0

            for age_type in age_types:
                count = 0
                df = filter_general_dynamics_dataframe(merged_dataframe, comparison_type, experiment, channel, age_type, None)
                for video_name in df['video_name'].unique():

                    df_this_video = df[(df['video_name']==video_name)]
                    table = df_this_video.pivot(index='frame', columns='particle', values=attribute)
                    y = table.mean(axis=1, skipna=True).reset_index()
                    y.columns = ['frame', 'mean']
                    if alignment=='cs': y['frame'] = y['frame'] - y['frame'].min()
                    elif alignment=='neb': y['frame'] = y['frame'] - df_this_video['man_neb'].unique()[0] + 1
                    elif alignment=='original': y['frame'] = y['frame']

                    if count==0: label = f'{comparison_type}: {experiment}, Channel: {channel}, Centrosoems: {age_type}'
                    else: label=None
                    axs[channel].plot(y['frame'], y['mean'], linestyle='-', marker='.', linewidth=0.5, c=line_colors_dict[age_type][channel], label=label)
                    count+=1

                    if np.min(y['frame']) < min_frame: min_frame = np.min(y['frame'])
                    if np.max(y['frame']) > max_frame: max_frame = np.max(y['frame'])
                    if np.max(y['mean']) > max_y: max_y = np.max(y['mean'])

            if alignment=='cs': 
                vxmin = neb
                left = 0
            elif alignment=='neb': 
                vxmin = 0
                left = min_frame
            else: 
                vxmin = None
                left = min_frame

            axs[channel] = set_line_plot(axs[channel], vxmin, max_frame, left, max_frame, max_y + max_y*0.2, ylabel=ylab_dict[attribute], channel=channel, ycolor=line_colors_dict[age_type][channel])

        fig.tight_layout()
        if len(age_types)==1: save_name = os.path.join(save_dir, f'{attribute}_{comparison_type}-{experiment}_alignment-{alignment}.png')
        elif len(age_types)==2: save_name = os.path.join(save_dir, f'omnm_{attribute}_{comparison_type}-{experiment}_alignment-{alignment}.png')
        fig.savefig(save_name, facecolor='white')
        plt.ioff()
        plt.close('all')



def batch_individual_embryo(input_dir, output_dir, attribute, label_age):
    ''' Generate graph of certain attribute for each embryo
    Args:
        input_dir: (str) the path that contains videos.csv and centrosomes.csv
        output_dir: (str) the directory where you want to save the figures. If None is present, it will point to the input_dir
        attribute: (str) e.g. 'total_intensity_norm' from centrosomes.csv column
        label_age: (bool) whether to show old_mother and new_mother
    '''
    dataframe_dict = read_data(input_dir, get_videos=True, get_centrosomes=True, get_models=True, get_simulations=True)
    dataframe_videos, dataframe_centrosomes, dataframe_models, dataframe_simulations = dataframe_dict['videos'], dataframe_dict['centrosomes'], dataframe_dict['models'], dataframe_dict['simulations']
    if (dataframe_videos is None) or (dataframe_centrosomes is None): return "quit batch_individual_embryo"
    if dataframe_models is None: residue_threshold = 1
    else: residue_threshold = dataframe_models['residual'].quantile(q=.9)

    if output_dir is None: output_dir = construct_folders(input_dir)["Individual"]
    save_dir = os.path.join(output_dir, attribute)
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    if label_age: age_types = ['new_mother', 'old_mother']
    else: age_types = ['all']

    for video_name in dataframe_videos['video_name'].unique():
        individual_embryo(dataframe_videos, dataframe_centrosomes, dataframe_simulations, dataframe_models, save_dir, video_name, attribute, age_types, residue_threshold)


def individual_embryo(dataframe_videos, dataframe_centrosomes, dataframe_simulations, dataframe_models, output_dir, video_name, attribute, age_types, residue_threshold):
    ''' Generate graph of certain attribute for an embryo
    Args:
        input_dir: (str) the path that contains videos.csv, centrosomes.csv, simulations.csv, models.csv
        output_dir: (str) the directory where you want to save the figures.
        video_name: (str) video_name of this videos
        attribute: (str) e.g. 'total_intensity_norm' from centrosomes.csv column
        age_types: (list) of (str) e.g.['old_mother', 'new_mother']
        residue_threshold: (float) the quality of the fitting
    '''
    df = dataframe_videos[(dataframe_videos['video_name']==video_name) & (dataframe_videos['channel']==0)]
    num_channel = dataframe_videos[dataframe_videos['video_name']==video_name]['num_channel'].unique()[0]
    cycle = df['cycle'] # cycle
    vxmin = df['s-phase_duration'].values[0] - 1
    line_colors_dict, cycle_dim_dict, ylab_dict = line_plot_styling()

    plt.style.use('ggplot')
    fig, ax0 = plt.subplots(figsize=(cycle_dim_dict[int(cycle)][1],5), dpi=100)
    if num_channel==1: axs = [ax0]
    elif num_channel==2: axs = [ax0, ax0.twinx()]

    for channel in range(num_channel):
        for age_type in age_types:
            x_original, y, y_err = mean_sd_centrosomes(dataframe_centrosomes, video_name, channel, attribute, age_type)
            x = np.arange(0, len(x_original), 1)
            residual = pd.Series({})

            if dataframe_models is not None:
                df = data_selection(dataframe_models, 'models', video_name, channel, age_type)
                attribute_conversion = {'total_intensity_norm':'intensity', 'area_um2':'area', 'mean_intensity_norm':'density', 'distance_um':'distance'}
                residual = df[df['attribute']==attribute_conversion[attribute]]['residual'].values

            if (dataframe_simulations is not None) and (residual.size > 0):
                df = data_selection(dataframe_simulations, 'simulations', video_name, channel, age_type)
                x_sim, y_sim = df['time'], df[attribute]
                if x_sim.any():
                    x_sim = x_sim.tolist()
                    start, end = np.argwhere(x_original==x_sim[0]), np.argwhere(x_original==x_sim[-1])
                    x_sim = x[start[0][0]:end[0][0] + 1]
                    warning_linestype, warning_color = '-', '#565656'
                    if residual[0] > residue_threshold: warning_linestype, warning_color = '--', '#8b0000'
                    axs[channel].plot(x_sim, y_sim, c=warning_color, linestyle=warning_linestype, linewidth=1, label='Fitted curve')

            axs[channel].plot(x, y, linestyle='-', marker='.', linewidth=0.5, c=line_colors_dict[age_type][channel], label=f'Channel: {channel}, Centrosoems: {age_type}')
            axs[channel].fill_between(x, y-y_err, y+y_err, color=line_colors_dict[age_type][channel], alpha=0.05)

            right = np.max(x)
            top = np.max((y+y_err + y_err*0.2))
            axs[channel] = set_line_plot(axs[channel], vxmin, right, 0, right, top, ylabel=ylab_dict[attribute], channel=channel, ycolor=line_colors_dict[age_type][channel])

    fig.tight_layout()
    if len(age_types)==1: save_name = os.path.join(output_dir, f'{attribute}_{video_name}.png')
    elif len(age_types)==2: save_name = os.path.join(output_dir, f'omnm_{attribute}_{video_name}.png')
    fig.savefig(save_name, facecolor='white')
    plt.ioff()
    plt.close('all')


def line_plot_styling():
    line_colors_dict = {'all':{0:'#009dff', 1:'#FF8C00'}, 'old_mother':{0:'#1338be', 1:'#FF8C00'}, 'new_mother':{0:'#009dff', 1:'#FFD700'}, 
                        'oom':{0:'#ff0f6b',1:'#ff0f6b'}, 'nom':{0:'#004cff',1:'#004cff'}, 'nm_oom':{0:'#26cf00',1:'#26cf00'}, 'nm_nom':{0:'#ff8800',1:'#ff8800'}}
    cycle_dim_dict = {11: (15, 4), 12:(15, 4), 13:(20, 5), 0:(40, 10)}
    ylab_dict = {'total_intensity_norm':'Total intensity (A.U.)', 'area_um2':'Area (um2)', 'mean_intensity_norm':'Mean intensity (A.U.)', 'distance_um':'Distance (um)',}
    return line_colors_dict, cycle_dim_dict, ylab_dict


def set_line_plot(ax, vxmin=None, vxmax=None, left=None, right=10, top=10, ylabel=None, channel=0, ycolor=None):
    ''' Set parameters for line plots
    Args:
        ax: (matplotlib.artist.Artist) a plot for drawing
        vxmin: (float) NEB time
        vxmax: (float) end time of this plot
        right: (float) last value of x-axis
        top: (float) last value of y-axis
        ylabel: (str) title of y-axis
        ycolor: (str) color of y-ticks
        label_legend: (bool) whether to shown legend
    Return:
        (matplotlib.artist.Artist) A drawn plots
    '''
    if vxmin is not None: ax.axvspan(xmin=vxmin, xmax=vxmax, facecolor='#ffff00', alpha=0.1) # Drawing the NEB time
    ax.set_ylim(bottom=0, top=top)
    ax.set_xlim(left=left, right=right)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    ax.tick_params(axis='y', labelcolor=ycolor)
    ax.set_ylabel(ylabel, fontsize='medium')
    ax.set_xlabel('Frame', fontsize='medium')
    if channel==0: 
        ax.legend(fontsize=8, loc=2)
        ax.tick_params(axis='both', labelsize='medium', top=False, bottom=True, left=True, right=False)
    else: 
        ax.legend(fontsize=8, loc=4)
        ax.tick_params(axis='both', labelsize='medium', top=False, bottom=True, left=False, right=True)
        ax.grid(None)
    ax.margins(x=0, tight=True)
    return ax


def filter_general_dynamics_dataframe(merged_dataframe, comparison_type=None, current_set=None, channel=0, age_type=None, particle_type=None):
    ''' Helper function to filer the dataframe specifically for ***general dynamics***
    '''
    if (age_type=='all') and (particle_type==None):
        cond = (merged_dataframe[comparison_type]==current_set)&(merged_dataframe['channel']==channel)
    elif (age_type!='all') and (particle_type==None):
        cond = (merged_dataframe[comparison_type]==current_set)&(merged_dataframe['channel']==channel)&(merged_dataframe['age_type']==age_type)
    elif (age_type=='all') and (particle_type!=None):
        cond = (merged_dataframe[comparison_type]==current_set)&(merged_dataframe['channel']==channel)&(merged_dataframe['particle_type']==particle_type)
    elif (age_type!='all') and (particle_type!=None):
        cond = (merged_dataframe[comparison_type]==current_set)&(merged_dataframe['channel']==channel)&(merged_dataframe['age_type']==age_type)&(merged_dataframe['particle_type']==particle_type)
    return merged_dataframe[cond]