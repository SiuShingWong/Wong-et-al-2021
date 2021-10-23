import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from centropy.analysis.pedigree import make_pedigree
from centropy.analysis.distance import pair_distance
from centropy.io import helper

def batch_centrosomes(input_dir, output_dir=None, tracking_program='TrackMate', main_channel=0, 
                      window_size=[10, 10], frame_rate=0.5, **kwargs):
    ''' Read videos.csv in input_dir and loop through videos and spots in tracks statistics.csv to 
        analyse and extract features from centrosomes. The output will save as centrosomes.csv and pedigree.csv 
    Args:
        input_dir: (str) directory containing videos.csv and all videos listed in videos.csv
        output_dir: (str) directory to save centrosomes.csv and pedigree.csv
        tracking_program: (str) 'TrackMate' or 'TrackPy'
        main_channel: (int) the channel to annotate the old mother / new mother
        window_size: (list of int) [x, y, z, ...] where x is the window size for channel 0, y for channel 1, ...
        frame_rate: (float) the duration of each frame e.g. 0.5 min per frame
        convert_dim: (bool) whether to convert the dimension of physical unit into pixel
        pix_scale: (float) the physical unit each pixel represents e.g. 0.11 means each pixel measure 0.11 um
        get_patch: (bool) whether to re-evaluate the centrosome features
        patch_method: (str) 'otsu', 'disk', 'yolo'
        get_pedigree: (bool) whether to obtain the pedigree of centrosomes
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
        pairing_frames: (int) The break between consecutive frames to be considered as a single division cluster
        get_cs: (bool) whether to annotate centrosome separation frame by this program
        cs_percentage: (float) at least this percentage of pair of centrosomes start to separate to classify the frame into cs frame
    '''

    if not output_dir: output_dir = input_dir

    get_patch = True
    get_pedigree = False
    convert_dim = False
    pix_scale = 0.11

    if 'get_patch' in kwargs: get_patch = kwargs['get_patch'] 
    if 'get_pedigree' in kwargs: get_pedigree = kwargs['get_pedigree'] 
    if 'convert_dim' in kwargs: convert_dim = kwargs['convert_dim']
    if 'pix_scale' in kwargs: pix_scale = kwargs['pix_scale']
    
    dataframe_videos = helper.read_data(input_dir, get_videos=True)['videos']
    if dataframe_videos is None: return "quite feature extraction"

    dataframe_videos = update_dataframe_videos(dataframe_videos, None, 'window_size', which_one=None, entry=window_size[1 - main_channel])
    dataframe_videos = update_dataframe_videos(dataframe_videos, input_dir, 'window_size', which_one=(dataframe_videos['channel']==main_channel), entry=window_size[main_channel], save=True)
    dataframe_videos_loop = dataframe_videos[dataframe_videos['analyze']=='yes'] # Only keep the videos with 'analyze' column set as 'yes'
    
    if tracking_program=='TrackMate': dataframe_centrosomes = centrosomes_from_trackmate(dataframe_videos, convert_dim, pix_scale)
    elif tracking_program=='TrackPy': dataframe_centrosomes = centrosomes_from_trackpy(dataframe_videos)
    else: return "please enter a tracking program"
    
    if get_patch:
        print ("quantifying centrosomes...")
        patch_method = 'otsu'
        if 'patch_method' in kwargs: patch_method = kwargs['patch_method'] 
        for _, row in tqdm(dataframe_videos_loop.iterrows()):
            video_name = row['video_name']
            video_path = row['video_path']
            channel = int(row['channel']) # channel of this video
            try: dataframe_centrosomes = centrosomes_features(dataframe_centrosomes, video_name, video_path, window_size[channel], channel, patch_method)
            except: print(f'cannot extract features from {video_name}'); continue
        dataframe_centrosomes.to_csv(os.path.join(output_dir, 'centrosomes.csv'), index=False)
    
    if get_pedigree:
        print ("identifying OM and NM...")

        max_pairing_distance = 50
        pairing_frames = 3
        cs_percentage = 0.75

        if 'max_pairing_distance' in kwargs: max_pairing_distance = kwargs['max_pairing_distance'] 
        if 'pairing_frames' in kwargs: pairing_frames = kwargs['pairing_frames'] 
        if 'cs_percentage' in kwargs: cs_percentage = kwargs['cs_percentage'] 
                        
        dataframe_videos = dataframe_videos[dataframe_videos['channel'] == main_channel]
        dataframe_pedigree = pd.DataFrame()
        for _, row in tqdm(dataframe_videos_loop.iterrows()):
            video_name = row['video_name']
            video_path = row['video_path']  # Video path from the video dataframe
            try:
                pedigree, dataframe_centrosomes = make_pedigree(dataframe_centrosomes, video_name, max_pairing_distance, pairing_frames, main_channel, False)
                dataframe_pedigree = pd.concat((dataframe_pedigree, pedigree))
            except: print(f'cannot extract pedigree from {video_name}'); continue

        dataframe_pedigree.to_csv(os.path.join(output_dir, 'pedigree.csv'), index=False)
        dataframe_videos['model_intensity_om'] = np.nan
        dataframe_videos['init_intensity_om'] = "_"
        dataframe_videos['model_intensity_nm'] = np.nan
        dataframe_videos['init_intensity_nm'] = "_"
    
    dataframe_centrosomes = normalize_centrosomes(dataframe_centrosomes, frame_rate, pix_scale)
    dataframe_centrosomes.sort_values(by=['video_name', 'channel', 'particle', 'frame'], inplace=True)
    dataframe_centrosomes.to_csv(os.path.join(output_dir, 'centrosomes.csv'), index=False)

    dataframe_videos.to_csv(os.path.join(output_dir, 'videos.csv'), index=False)


def update_videos_path(dataframe_videos, root_dir=None, save=False):
    ''' Update the video paths of every videos in videos.csv
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        root_dir: (str) directory of videos.csv and all videos in tif format
        save: (bool) whether to save the updated videos.csv
    Return:
        updated paths in videos.csv 
    '''
    from pathlib import Path
    
    for i, row in dataframe_videos.iterrows():
        video_name = row['video_name']
        num_channel = row['num_channel']

        if num_channel==1:
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if (video_name in f) and f.endswith('.tif'):
                        dataframe_videos.loc[dataframe_videos['video_name']==video_name, 'video_path'] = os.path.join(root, f)
                        dataframe_videos.loc[dataframe_videos['video_name']==video_name, 'nucleus_path'] = os.path.join(root, f[:-4] + '_nucleus.tif')
        
        else:
            channel = row['channel']
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if (Path(video_name).stem in f) and (f.endswith('.tif')) and (f'C{channel+1}' in f) and ('nucleus' not in f):
                        cond = (dataframe_videos['video_name']==video_name)&(dataframe_videos['channel']==channel)
                        dataframe_videos.loc[cond, 'video_path'] = os.path.join(root, f)
                        dataframe_videos.loc[cond, 'nucleus_path'] = os.path.join(root, f[:-4] + '_nucleus.tif')
        
    if save: dataframe_videos.to_csv(os.path.join(root_dir, 'videos.csv'), index=False)
    return dataframe_videos


def update_dataframe_videos(dataframe_videos, root_dir=None, column=None, which_one=None, entry=None, save=False):
    ''' Apply changes to columns in videos.csv
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        root_dir: (str) directory where videos.csv and all videos are
        column: (str) column in videos.csv or column to be added
        which_one: logical to extract certain row
        entry: (str) value to be added
    Return:
        updated videos.csv 
    '''
    if root_dir!=None:
        # dataframe_videos_path = os.path.join(root_dir, 'videos.csv')
        dataframe_videos = update_videos_path(dataframe_videos, root_dir, save)
    
    if (column!=None) and (entry!=None):
        if column not in dataframe_videos.columns: dataframe_videos[column] = np.nan
        if which_one is None: dataframe_videos[column] = entry
        else: dataframe_videos.loc[which_one, column] = entry
    
    if save: dataframe_videos.to_csv(os.path.join(root_dir, 'videos.csv'), index=False)
    return dataframe_videos


def centrosomes_from_trackmate(dataframe_videos, convert_dim=False, pix_scale=1):
    ''' Construct a primitive centrosomes.csv from all Spots in track statistics.csv generated by TrackMate
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        convert_dim: (bool) whether to convert the dimension of physical unit into pixel
        pix_scale: (float) the physical unit each pixel represents e.g. 0.11 means each pixel measure 0.11 um
    Return:
        centrosomes.csv
    '''
    dataframe_centrosomes = pd.DataFrame()
    for _, row in dataframe_videos.iterrows():
        
        video_name = row['video_name']
        video_path = row['video_path']
        channel = row['channel']
        
        spots_in_track_statistics_path = os.path.join('/'.join(video_path.split('/')[:-1]), 'Spots in tracks statistics.csv')
        if not os.path.exists(spots_in_track_statistics_path): print (f'spots_in_track_statistics.csv not found in {video_path}'); continue
        
        df = initialize_centrosomes(video_name, spots_in_track_statistics_path, channel)
        if convert_dim: df['x'], df['y'] = df['x']/pix_scale, df['y']/pix_scale # if x, y coordinates are not in pixel unit
        dataframe_centrosomes = pd.concat((dataframe_centrosomes, df))

    dataframe_centrosomes = dataframe_centrosomes.reset_index(drop=True)
    dataframe_centrosomes['x'], dataframe_centrosomes['y'] = dataframe_centrosomes['x'].astype('int16'), dataframe_centrosomes['y'].astype('int16')
        
    return dataframe_centrosomes


def initialize_centrosomes(video_name, spots_in_track_statistics_path, channel=0):
    ''' Initialize centrosomes.csv with columns renamed from Spots in tracks statistics.csv
    Args:
        video_name: (str) video name of this pedigree
        spots_in_track_statistics_path: (str) path to Spots in tracks statistics.csv of each video
        channel: (int) channel
    Return:
        centrosomes.csv
    '''
    df_in = pd.read_csv(spots_in_track_statistics_path)  # dataframe from the path for obtaining information
    df = pd.DataFrame(columns=[ 'video_name', 'particle', 'tracking', 'x', 'y', 'frame', 'channel',
                                'total_intensity', 'area', 'mean_intensity', 'pair_id', 'age_type',
                                'distance', 'particle_type',])  # Create a dataframe with some columns
    try:
        df['particle'] = df_in['TRACK_ID']
        df['x'] = df_in['POSITION_X']
        df['y'] = df_in['POSITION_Y']
        df['frame'] = df_in['FRAME']
        df['channel'] = channel
        df['video_name'] = video_name
        df['total_intensity'] = df_in['TOTAL_INTENSITY']
        df['mean_intensity'] = df_in['MEAN_INTENSITY']
        df['area'] = df_in['RADIUS']
    except: print ('initialize_centrosomes has errors!')
    return df


def centrosomes_features(dataframe_centrosomes, video_name=None, video_path=None, window_size=28, channel=0, method='otsu'):
    ''' Extract features of centrosomes from a ** single video ** e.g. area, intensity, ...
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        video_name: (str) string of this video's name
        video_path: (str) path to the video of video_name
        window_size: (int) size of bounding box to calculate features
        channel: (int) channel of this video
        method: (str) 'otsu', 'disk', 'yolo'
    Return:
        updated data_centrosomes with added columns of features
    '''
    from skimage.io import imread
    from skimage.filters import threshold_otsu
    
    cond_this = (dataframe_centrosomes['video_name']==video_name) & (dataframe_centrosomes['channel']==channel)
    this_dataframe_centrosomes = dataframe_centrosomes[cond_this]
    video = imread(video_path)
    
    if method=='otsu': otsu_threshold = int(threshold_otsu(video[int(this_dataframe_centrosomes['frame'].min()),:,:]))
    if len(video.shape) == 3:  _, height, width = video.shape 
    else: return 'Not a matrix with time x height x width'
    
    # Loop through all centrosomes in this video
    for _, row in this_dataframe_centrosomes.iterrows():
        particle = row['particle']  # this particle
        frame = row['frame']  # this frame
        x, y = row['x'], row['y']  # X, Y position of this centrosome
        xmin, ymin, xmax, ymax = x-window_size//2, y-window_size//2, x+window_size//2, y+window_size//2
        
        if method=='yolo':
            try:
                w, h = row['w'], row['h']
                xmin, ymin, xmax, ymax = x-w//2, y-h//2, x+w//2, y+h//2
            except: print ('w and h are not found. Please use another segmentation method.')
            
        if not (width>xmin>0 and height>ymin>0 and width>xmax>0 and height>ymax>0): continue  # The the particle is near the edge of the video, it will be discarded
        
        patch = video[frame, ymin:ymax, xmin:xmax]
        if method=='otsu': mask, skip_patch = threshold_mask(patch, otsu_threshold) # Otsu threshold
        elif method=='disk': mask, skip_patch = disk_mask(patch) # A circular ROI within the window
        elif method=='yolo': mask, skip_patch = rectangle_mask(patch) # For Yolo manually annotated particles
        elif method=='rectangle': mask, skip_patch = rectangle_mask(patch) # For Yolo manually annotated particles
        else: continue
        if skip_patch: continue 
        
        area, total_intensity, mean_intensity = patch_features(patch, mask)
        cond = cond_this & (dataframe_centrosomes['particle'] == particle) & (dataframe_centrosomes['frame'] == frame)

        dataframe_centrosomes.loc[cond, 'total_intensity'] = total_intensity
        dataframe_centrosomes.loc[cond, 'area'] = area
        dataframe_centrosomes.loc[cond, 'mean_intensity'] = mean_intensity
            
    return dataframe_centrosomes


def threshold_mask(patch, otsu_threshold):

    from scipy.spatial import distance
    from skimage.measure import label, regionprops
    
    patch_bin = patch > otsu_threshold
    patch_lab = label(patch_bin)  # label the different region of the patch
    patch_props = regionprops(patch_lab)  # calculate the region within the patch
    
    if len(patch_props) == 0:
        return 0, True
    
    if len(patch_props) == 1:  # If just one region is found, it's the best regions
        mask = patch_bin.astype(np.int8)
        return mask, False
        
    window_size = patch.shape[0]
    min_dist = 200
    for r in patch_props:
        dist = distance.euclidean((r.centroid[0], r.centroid[1]), (window_size // 2, window_size // 2))
        if dist < min_dist:  # The region closest to the center is defined as the best region
            min_dist = dist
            best_reg = r
            label = r.label
            
    mask = (patch_lab == label).astype(np.int8)
    return mask, False


def disk_mask(patch):
    from skimage.draw import disk
    
    window_size = min(patch.shape[0], patch.shape[1])
    mask = np.zeros_like(patch)
    rr, cc = disk((window_size//2, window_size//2), window_size//2, shape=mask.shape)

    mask[rr, cc] = 1
    return mask, False


def rectangle_mask(patch):
    
    mask = np.zeros_like(patch)
    mask[:, :] = 1
    
    return mask, False


def patch_features(patch, mask):
    
    masked_patch = patch * mask
    
    area = mask.sum()
    total_intensity = masked_patch.sum()
    mean_intensity = np.nanmean(np.where(masked_patch != 0, masked_patch, np.nan))

    return area, total_intensity, mean_intensity


def normalize_centrosomes(dataframe_centrosomes, frame_rate=0.5, pix_scale=0.11):
    ''' Normalize different feature columns of centrosomes.csv e.g. distance, area, ...
    Args:
        dataframe_centrosomes: (pandas.DataFrame) feature-extracted centrosomes.csv i.e. area, intensity, .. columns are filled
        frame_rate: (float) the duration of each frame e.g. 0.5 min per frame
        pix_scale: (float) the physical unit each pixel represents e.g. 0.11 means each pixel measure 0.11 um
    Return:
        updated data_centrosomes with updated columns mapped to physical unit and normalized
    '''
    from centropy.analysis.dataframe import convert_frame2time
    from centropy.analysis.dataframe import rescale
    
    if 'frame' in dataframe_centrosomes.columns:
        dataframe_centrosomes = convert_frame2time(dataframe_centrosomes, frame_rate)
    if 'area' in dataframe_centrosomes.columns:
        dataframe_centrosomes['area_um2'] = dataframe_centrosomes['area'] * pix_scale**2
        dataframe_centrosomes = rescale(dataframe_centrosomes, 'area', 'mean')
    if 'distance' in dataframe_centrosomes.columns:
        dataframe_centrosomes['distance_um'] = dataframe_centrosomes['distance'] * pix_scale
        dataframe_centrosomes = rescale(dataframe_centrosomes, 'distance', 'mean')
    if 'total_intensity' in dataframe_centrosomes.columns:
        dataframe_centrosomes = rescale(dataframe_centrosomes, 'total_intensity', 'mean')
    if 'mean_intensity' in dataframe_centrosomes.columns:
        dataframe_centrosomes['mean_intensity_norm'] = dataframe_centrosomes['total_intensity_norm'] / dataframe_centrosomes['area_norm']
    
    return dataframe_centrosomes
