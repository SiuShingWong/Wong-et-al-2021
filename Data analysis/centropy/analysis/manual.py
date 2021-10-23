import os

import numpy as np
import pandas as pd

from skimage.io import imread
import trackpy as tp

from centropy.analysis.dynamics import update_dataframe_videos, centrosomes_features, normalize_centrosomes

def batch_manual(input_dir, output_dir=None, main_channel=0, window_size=[10, 10], frame_rate=0.5, **kwargs):
    ''' Read videos.csv in input_dir and loop through videos to track, analyse, and extract features from centrosomes.
        The output will save as manual.csv
    Args:
        input_dir: (str) directory containing videos.csv and all videos listed in videos.csv
        output_dir: (str) directory to save centrosomes.csv and pedigree.csv
        main_channel: (int) the channel to track
        window_size: (list of int) [x, y, z, ...] where x is the window size for channel 0, y for channel 1, ...
        frame_rate: (float) the duration of each frame e.g. 0.5 min per frame
        linking_distance: (int) distance threshold to link centrosomes across frames
        convert_dim: (bool) whether to convert the dimension of physical unit into pixel
        pix_scale: (float) the physical unit each pixel represents e.g. 0.11 means each pixel measure 0.11 um
        get_patch: (bool) whether to re-evaluate the centrosome features
        patch_method: (str) 'otsu', 'disk', 'yolo'
    '''
    if not output_dir: output_dir = input_dir

    linking_distance = 50
    get_patch = True
    pix_scale = 0.11

    if 'linking_distance' in kwargs: linking_distance = kwargs['linking_distance'] 
    if 'get_patch' in kwargs: get_patch = kwargs['get_patch'] 
    if 'pix_scale' in kwargs: pix_scale = kwargs['pix_scale']

    try: dataframe_videos = pd.read_csv(os.path.join(input_dir, 'videos.csv')) 
    except: return "could not find videos.csv"

    # Updating videos.csv
    dataframe_videos = update_dataframe_videos(dataframe_videos, None, 'window_size', which_one=None, entry=window_size[1 - main_channel])
    dataframe_videos = update_dataframe_videos(dataframe_videos, None, 'window_size', which_one=(dataframe_videos['channel']==main_channel), entry=window_size[main_channel])
    dataframe_videos = dataframe_videos[dataframe_videos['analyze']=='yes'] # Only keep the videos with 'analyze' column set as 'yes'

    # Obtaining tracks using TrackPy
    dataframe_manual = manual_from_trackpy(dataframe_videos, main_channel, linking_distance)

    # Quantify the spots from TrackPy
    if get_patch:

        print ("quantifying centrosomes...")
        patch_method = 'yolo'
        if 'patch_method' in kwargs: patch_method = kwargs['patch_method']
        for _, row in dataframe_videos.iterrows():
            video_name = row['video_name']
            video_path = row['video_path']
            channel = int(row['channel']) # channel of this video
            
            dataframe_manual = centrosomes_features(dataframe_manual, video_name, video_path, window_size[channel], channel, patch_method)
            # except: print(f'cannot extract features from {video_name}'); continue

    dataframe_manual = normalize_centrosomes(dataframe_manual, frame_rate, pix_scale)
    dataframe_manual.sort_values(by=['video_name', 'channel', 'particle', 'frame'])
    dataframe_manual.to_csv(os.path.join(output_dir, 'manual.csv'), index=False)


def manual_from_trackpy(dataframe_videos, main_channel=0, linking_distance=50):
    ''' Construct a primitive manual.csv using TrackPy
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        main_channel: (int) the channel to track
        linking_distance: (int) distance threshold to link centrosomes across frames
    Return:
        manual.csv
    '''
    dataframe_manual = pd.DataFrame()
    dataframe_videos_tracking = dataframe_videos[dataframe_videos['channel']==main_channel] # Keep the videos of the channel for tracking
    for _, row in dataframe_videos_tracking.iterrows():
        video_name = row['video_name']
        video_path = row['video_path']
        channel = row['channel']

        df = initialize_manual(video_name, video_path, channel, linking_distance) # Tracking
        dataframe_manual = dataframe_manual.append(df)

    dataframe_videos_tracking = dataframe_videos[dataframe_videos['channel']!=main_channel] # Keep the videos of the channel not for tracking
    for _, row in dataframe_videos_tracking.iterrows():
        video_name = row['video_name']
        channel = row['channel']

        df = dataframe_manual[dataframe_manual['video_name']==video_name].copy()
        df.loc[:, 'channel'] == channel
        dataframe_manual.append(df)

    dataframe_manual.sort_values(by=['video_name', 'channel', 'particle', 'frame'], inplace=True)
    return dataframe_manual


def initialize_manual(video_name, video_path, channel, linking_distance):
    ''' Initialize manual.csv from TrackPy's output
    Args:
        video_name: (str) video name of this pedigree
        video_path: (str) path to the video of video_name
        channel: (int) channel
        linking_distance: (int) distance threshold to link centrosomes across frames
    Return:
        manual.csv
    '''

    video = imread(video_path) # Read the video
    nframe, nrows, ncols = video.shape
    
    root = '/'.join(video_path.split('/')[:-1])
    label_path = os.path.join(root, 'label/YOLO_darknet') # The location of YOLO files
    if not os.path.exists(label_path): print(f'could not find YOLO files in {video_path}. Please annotate using OpenLabelling'); return pd.DataFrame()
    frame_dict = yolo_from_file(label_path) # A dictionary of frame : path

    df = pd.DataFrame()
    for frame in range(nframe):
        yolo_boxes = read_yolo_boxes(frame_dict[frame])
        if len(yolo_boxes) > 0: # If there's annotation in this frame
            particle_type = yolo_boxes[:,0].astype(np.int)
            yolo_x, yolo_y, yolo_w, yolo_h = yolo_boxes[:,1], yolo_boxes[:,2], yolo_boxes[:,3], yolo_boxes[:,4]
            x, y, w, h = (yolo_x*ncols).astype(int), (yolo_y*nrows).astype(int), (yolo_w*ncols).astype(int), (yolo_h*nrows).astype(int)
            df = df.append(pd.DataFrame({'video_name':video_name, 'channel':channel, 'frame':frame, 'x':x, 'y':y, 'w':w, 'h':h, 
                                         'particle_type':particle_type, 'yolo_x':yolo_x, 'yolo_y':yolo_y, 'yolo_w':yolo_w, 'yolo_h':yolo_h}))
    if 'frame' not in df.columns: print(f'no frame is found in {video_name}'); return df
        
    df.reset_index(inplace=True, drop=True)
    tp.quiet(); df = tp.link(df, linking_distance, memory=3) # Tracking
    return df


def yolo_from_file(infolder, ext='.txt'):
    ''' Read text files of YOLO format generated from OpenLabeling 
    '''
    frame_dict = {}
    for root, dirs, files in os.walk(infolder):
        for names in files:
            if names.endswith(ext) or names.endswith(ext.upper()):
                stem_name = os.path.splitext(names)[0]
                frame = int(stem_name.split('_')[-1])
                frame_dict[frame] = os.path.join(root, names)
    return frame_dict


def read_yolo_boxes(filepath):
    ''' Convert YOLO format from string to the corresponding number
    '''
    box = []
    with open(filepath,'r') as f:
        for line in f:
            box_info = (np.hstack(line.strip().split())).astype(np.float)
            box.append(box_info)
    if len(box) > 0:
        return np.vstack(box)
    else:
        return []