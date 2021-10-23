import numpy as np
import pandas as pd

def rescale(dataframe_centrosomes, attribute, method='mean'):
    ''' Rescale of attribute from raw data values to adjusted value by find the minimum of the means of centrosome dynamics
        and normalize to that in each channel
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        attribute: (str) can be 'area', 'total_intensity', ...
    Return:
        (pandas.DataFrame) updated centrosomes.csv with an added 'attribute_norm' column where
        attribute is the same as that in argument
    '''
    colname = f'{attribute}_norm'
    dataframe_centrosomes[colname] = np.nan # initialization
    
    for channel in dataframe_centrosomes['channel'].unique():
        cond_channel = dataframe_centrosomes['channel']==channel
        attribute_min = dataframe_centrosomes[cond_channel][attribute].max()

        for video_name in dataframe_centrosomes['video_name'].unique():
            cond_video = dataframe_centrosomes['video_name']==video_name
            table = dataframe_centrosomes[cond_video & cond_channel].pivot(index='frame', columns='particle', values=attribute)
            attribute_val = table.mean(axis=1).values.min()
            
            if attribute_val < attribute_min:  attribute_min = attribute_val
                
        dataframe_centrosomes.loc[cond_channel, colname] = dataframe_centrosomes[cond_channel][attribute] / attribute_min # normalization to the minimum of means

    return dataframe_centrosomes


def convert_frame2time(dataframe_centrosomes, frame_rate=0.5):
    ''' Convert frame of centrosomes.csv of each videos into time, so that the first timepoint is 0
    Args:
        dataframe_centrosomes: (pandas.DataFrame) feature-extracted centrosomes.csv i.e. area, intensity, .. columns are filled
        frame_rate: (float) the duration of each frame e.g. 0.5 min per frame
    '''
    dataframe_centrosomes['time'] = np.nan
    for video_name in dataframe_centrosomes['video_name'].unique():
        df = dataframe_centrosomes[dataframe_centrosomes['video_name'] == video_name]
        dataframe_centrosomes.loc[dataframe_centrosomes['video_name'] == video_name, 'time'] = (df['frame'] - df['frame'].min()) * frame_rate
    return dataframe_centrosomes


def filter_merge_videos_prism(dataframe_videos, dataframe_any, res_threshold=0.1, frame_rate=0.5):
    ''' Merge videos.csv and other data with filtered resdiual, and time conversion for Prism
    Args:
        data_videos: (pandas.DataFrame) videos.csv
        data: (pandas.DataFrame) can be centrosomes.csv, models.csv, ...
        res_threshold: (float) fitting error threshold
        frame_rate: (float) Conversion ratio from frame to time
    Return:
        pandas.DataFrame of the merged and filtered dataframe
    '''
    # if 'channel' in data.columns:
    #     data_videos = data_videos.drop(columns='channel')
    df = pd.merge(dataframe_videos, dataframe_any, on=['video_name', 'channel']) # merge them based on column 'video_name'
    if 's-phase_duration' in df.columns: df['s-phase_duration'] = df['s-phase_duration']*frame_rate # change the unit of s-phase_duration from frame to minutes
    if 'decrease_rate_m' in df.columns: df['decrease_rate_m'] = df['decrease_rate_m'].abs() # change the sign of decrease_rate_m to positive
    if 'age_type' in df.columns: df['age_type'] = df['age_type'].astype('category') # change the type of the column 'age_type' to category
    if ('residual' in df.columns) and (res_threshold is not None): df = df[df['residual'] < res_threshold] # remove data with fitting residual > RES_THRESHOLD
    return df


def filter_merge_videos(dataframe_videos, dataframe_any, res_threshold=0.1, frame_rate=0.5, compare_attribute=False):
    ''' Merge videos.csv and other data with filtered resdiual, and time conversion
    Args:
        data_videos: (pandas.DataFrame) videos.csv
        data: (pandas.DataFrame) can be centrosomes.csv, models.csv, ...
        res_threshold: (float) fitting error threshold
        frame_rate: (float) Conversion ratio from frame to time
        compare_attribute: (bool) Whether to compare attribute e.g. compare 'area' to 'intensity', ...
    Return:
        pandas.DataFrame of the merged and filtered dataframe
    '''

    df = pd.merge(dataframe_videos, dataframe_any, on=['video_name', 'channel']) # merge them based on column 'video_name'
    if 'cycle' in df.columns:
        if df['cycle'].any(): df['cycle'] = df['cycle'].astype('category') # change the type of the column 'cycle' to category
    if 'manipulated_protein' in df.columns:
        if df['manipulated_protein'].any(): df['manipulated_protein'] = df['manipulated_protein'].astype('category') # change the type of the column 'manipulated_protein' to category
    if 's-phase_duration' in df.columns: df['s-phase_duration'] = df['s-phase_duration']*frame_rate # change the unit of s-phase_duration from frame to minutes
    if 'decrease_rate_m' in df.columns: df['decrease_rate_m'] = df['decrease_rate_m'].abs() # change the sign of decrease_rate_m to positive
    if 'age_type' in df.columns: df['age_type'] = df['age_type'].astype('category') # change the type of the column 'age_type' to category
    if ('residual' in df.columns) and (res_threshold is not None):
        df = df[df['residual'] < res_threshold] 
    if 'attribute' in df.columns:
        if compare_attribute: df['attribute'] = df['attribute'].astype('category')
    if 'channel' in df.columns: df['channel'] = df['channel'].astype('category')
    return df


def mean_sd_centrosomes(dataframe_centrosomes, video_name=None, channel=0, attribute=None, age_type=None):
    '''Get x, y, sd from centrosomes.csv under these attribute
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        video_name: (str) video name of a single video
        channel: (int) 0 or 1
        attribute: (str) 'total_intensity_norm', 'mean_intensity_norm', 'area_um2', 'distance_um'
        age_type: (str) 'all', 'old_mother', 'new_mother'
    Return:
        x, y, sd: (1D numpy.array)
    '''
    if age_type == 'all': table = data_selection(dataframe_centrosomes, 'centrosomes', video_name, channel, age_type).pivot(index='time', columns='particle', values=attribute)
    else: table = data_selection(dataframe_centrosomes, 'centrosomes', video_name, channel, age_type).pivot(index='time', columns='particle', values=attribute)
    return table.index.values, table.mean(axis=1, skipna=True).values, table.std(axis=1, skipna=True).values


def data_selection(dataframe, data_type=None, video_name=None, channel=0, age_type=None):
    '''Select specific data based on video name, channel, age type
    Args:
        dataframe: (pandas.DataFrame) centrosomes.csv, models.csv, simulations.csv
        data_type: (str) 'centrosomes', 'models', 'simulations'
        video_name: (str) video name of a single video
        channel: (int) 0 or 1
        age_type: (str) 'all', 'old_mother', 'new_mother'
    Return:
        results: (pandas.DataFrame) selected video
    '''
    cond_video = dataframe['video_name']==video_name
    cond_channel = dataframe['channel']==channel
    if data_type == 'centrosomes':
        if age_type=='all': df = dataframe[cond_video & cond_channel]
        else: df = dataframe[cond_video & cond_channel & (dataframe['age_type']==age_type)]
    else:
        df = dataframe[cond_video & cond_channel & (dataframe['age_type']==age_type)]
    return df