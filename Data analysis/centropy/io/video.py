import os
from pathlib import Path
import datetime

import numpy as np
import pandas as pd

from skimage.io import imsave, imread

def batch_process_videos(input_dir, output_dir, processed):
    ''' Process and enter the meta-data of a video into videos.csv
    Args:
        input_dir: (str) directory containing all videos
        output_dir: (str) directory storing the dataset
        processed: (bool) whether videos have been processed
    '''
    if processed:
        root_dir = input_dir
        save_dir = input_dir # If it has been processed, just save it in the input directory
    else:
        root_dir = os.path.join(output_dir, input_dir.split('/')[-1]) # a directory to store the dataset using the name of input_dir
        save_dir = os.path.join(root_dir, 'Videos')
        if not os.path.exists(root_dir): os.mkdir(root_dir)
        if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    dataframe_videos_path = os.path.join(root_dir, 'videos.csv')

    if os.path.exists(dataframe_videos_path):
        dataframe_videos = pd.read_csv(dataframe_videos_path)
        logged_videos = dataframe_videos['video_name'].unique().tolist()
    else:
        dataframe_videos = initialize_videos()
        logged_videos = []
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            if (f not in logged_videos) and (f.endswith('.tif')) and ('nucleus' not in f):
                if not processed: 
                    df = process_video(os.path.join(root, f), save_dir)
                    dataframe_videos = dataframe_videos.append(df)
                else:
                    df = enter_video_info(os.path.join(root, f))
                    dataframe_videos = dataframe_videos.append(df)
    
    dataframe_videos.sort_values(by=['video_name', 'channel'], inplace=True)
    dataframe_videos.to_csv(os.path.join(root_dir, 'videos.csv'), index=False)


def initialize_videos():
    """ Make DataFrame object contain the following fields in videos.csv
    Return:
        df: (pandas.DataFrame) empty dataframe
    """
    df =  pd.DataFrame(columns = ['datetime', 'video_name', 'cycle', 'genotype', 'manipulated_protein', 'manipulation',
                'num_frame', 'num_stack', 'num_channel', 'channel', 'width', 'height', 'raw_video_path', 'video_path', 'nucleus_path', 
                'tracking_program', 'blob_size', 'init_quality', 'quality', 'man_cs', 'man_neb', 's-phase_duration', 'remarks',
                'image_preprocessing', 'exp_correction_A', 'exp_correction_B', 'max_intensity', 'min_intensity', 
                'model_intensity_all', 'init_intensity_all', 'analyze', 'avi_output'])
    return df


def process_video(video_path, output_dir):
    ''' Process a **single video** and add metadata to viedos.csv
    Args:
        video_path: (str) path to a single video
        output_dir: (str) path of the final destination of videos
    Return:
        df: (pandas.DataFrame) information of the current video
    '''
    video = imread(video_path)
    name_stem = Path(video_path).stem # The name of video without the extension
    processed_video_path = os.path.join(output_dir, name_stem) # Directory to store the processed video
    if not os.path.exists(processed_video_path): os.mkdir(processed_video_path)

    if len(video.shape)==4: num_channel = 1
    elif len(video.shape)==5: num_channel = video.shape[2]

    df = pd.DataFrame()
    for channel in range(num_channel):

        if num_channel == 1: 
            save_name = name_stem
            video_channel = video
        elif num_channel > 1: 
            save_name = name_stem + f'_C{channel + 1}'
            video_channel = video[:,:,channel,:,:]
        else: 
            print (f'{name_stem}.tif has the shape of {video.shape}')
            return df
        nucleus_save_name = save_name + '_nucleus'

        num_frame, num_stack, height, width = video_channel.shape
        processed_video, exponential_decay = preprocess_video(video_channel)

        imsave(os.path.join(processed_video_path, save_name + '.tif'), processed_video)
        imsave(os.path.join(processed_video_path, nucleus_save_name + '.tif'), video_channel[:, -1, :, :])

        min_intensity, max_intensity = processed_video.min(), processed_video.max()
        image_preprocessing_pipeline = 'exponential_decay_correction > uneven_illumination_correction > maximum_intensity_projection'

        df = df.append({'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'video_name':video_path.split('/')[-1], 'num_frame': num_frame, 'num_stack': num_stack, 'num_channel': num_channel,
                        'channel': channel, 'width': width, 'height': height, 'video_path': os.path.join(processed_video_path, save_name + '.tif'), 
                        'nucleus_path': os.path.join(processed_video_path, nucleus_save_name + '.tif'), 'raw_video_path': video_path, 
                        'tracking_program': 'TrackMate', 'image_preprocessing': image_preprocessing_pipeline, 
                        'exp_correction_A': exponential_decay[0], 'exp_correction_B': exponential_decay[1], 'max_intensity': max_intensity,
                        'min_intensity': min_intensity, 'model_intensity_all':'linear_plateau', 'init_intensity_all':'_', 
                        'analyze':'yes', 'avi_output':'general', }, ignore_index=True)
    
    return df


def preprocess_video(vid):
    """ Perform preprocessing (Exponential decay correction, Uneven illumination correction, 
    maximum intensity projection) of a movie (np.array). It employs multiprocessing to speed up the process. 
    It return a preprocessed movie (np.array)
    Args:
        vid: 4D np.array (time x depth x height x width)
    Return:
        3D np.array (time x height x width), and exponential decay parameters
    """
    vid_temp, params = exponential_decay_correction(vid.copy())

    t, _, _, _ = vid.shape

    out_vid = []
    for i in range(t):
        temp = correct_uneven_illumination(vid_temp[i, :, :, :])
        out_vid.append(max_intensity_projection(temp))

    return np.stack(out_vid, axis=0), params


def exponential_decay_correction(vid):
    """ Perform exponential decay correction and return the corrected video 
    Args:
        vid: 4D np.array of time x depth x height x width
    return
        4D np.array of time x depth x height x width and parameters
    """
    vid_corrected, (baseline, params) = remove_exponential_vid_decay(vid)
    return vid_corrected.astype(vid.dtype), params


def remove_exponential_vid_decay(vid, channel=0, average_fnc=None, f_scale=100.):
    """
    Fits an equation of form y=Ae^(Bt)+C on the mean intensities of video
    """
    from scipy.stats import linregress
    from scipy.optimize import least_squares
    
    if average_fnc is None:
        average_fnc = np.mean
    if len(vid.shape) == 4:
        vid_in = vid[:, :, :, channel]

    I_vid = np.hstack([average_fnc(v) for v in vid])
    I_time = np.arange(len(I_vid))
    
    # fit equation. y =A*e^(-Bt)
    log_I_vid = np.log(I_vid)
    slope, intercept, r_value, p_value, std_err = linregress(I_time, log_I_vid)

    # initial fit. 
    A = np.exp(intercept)
    B = slope
    
    # refined robust fitting. 
    def exp_decay(t,x):
        return (x[0] * np.exp(x[1] * t) + x[2])
        
    def res(x, t, y):
        return exp_decay(t,x) - y
    
    x0 = [A, B, 0]
    try:
        res_robust = least_squares(res, x0, loss='soft_l1', f_scale=f_scale, args=(I_time, I_vid))
            
        robust_y_baseline = exp_decay(I_time, res_robust.x)
        correction = float(robust_y_baseline[0]) / (robust_y_baseline + .1)
        
        vid_corrected = np.zeros(vid.shape, np.float32)
        
        for frame in range(vid.shape[0]):
            vid_corrected[frame, ...] = vid[frame, ...] * correction[frame]
        
        return vid_corrected, (robust_y_baseline, res_robust.x)
    except ValueError:
        return vid, ('NIL', 'NIL')


def correct_uneven_illumination(img, ksize=5, presmooth=False, downscale=8, background='light'):
    
    # from skimage.transform import resize
    # from skimage.morphology import black_tophat, white_tophat, disk, square, opening, closing
    from skimage.filters import gaussian
    # from skimage.exposure import rescale_intensity

    im_ = img.copy()  # make internal copy
    dtype = type(im_.ravel()[0])  # infer datatype.

    if len(im_.shape) == 3:
        im_ = im_[..., None]

    im_out = np.zeros_like(im_)  # return same type.
    d, h, w, c = im_.shape

    for ii in range(d):
        for jj in range(c):
            im_slice = im_[ii, :, :, jj]
            if presmooth:
                im_slice = gaussian(im_slice, sigma=1)

            im_bg = estimate_background(im_slice, ksize=ksize, downscale=downscale, background=background)

            if background == 'light':
                # im_slice = white_tophat(im_slice, disk(ksize))
                corrected = (im_slice - im_bg)
                # corrected = corrected + np.mean(im_bg)
                im_slice = np.clip(corrected, 0, np.inf).astype(dtype)
            if background == 'dark':
                # im_slice = black_tophat(im_slice, disk(ksize))
                corrected = (im_bg - im_slice)
                # corrected = np.mean(im_bg) - corrected
                im_slice = np.clip(corrected, 0, np.inf).astype(dtype)

            # write out.
            im_out[ii, :, :, jj] = im_slice.copy()

    return np.squeeze(im_out)


def estimate_background(img, ksize=5, downscale=4, background='light'):
    from skimage.morphology import opening, closing, disk
    from skimage.transform import rescale, resize

    # downsample
    im_down = rescale(img, 1. / downscale, preserve_range=True)

    if background == 'light':
        im_down = opening(im_down, disk(ksize))
    if background == 'dark':
        im_down = closing(im_down, disk(ksize))

    # upsample
    im_bg = resize(im_down, img.shape, preserve_range=True)

    return im_bg


def max_intensity_projection(img):
    """ Perform the maximum intensity projection of 4D numpy array (but choose a particular channel)
    into a 2D numpy array.

    For each entry of the matrix, it will compare the value of the same entry in another depth. It returns
    the one with the highest value across all depth.

    Args:
        img: 4D np.array dimension: depth x height x width x channel
        channel: integer indicating which plane you wish to perform the maxumum intensity projection

    Return:
        2D np.array storing intensity value of the max projection on a selectede channel
    """

    d, h, w = img.shape  # Retrieve the dimension of the image
    ret_img = np.zeros((h, w))  # Inititalize a zero array for the return image
    temp_img = np.zeros((h, w))  # Initialize a zero array to temporarily store the image data

    for i in range(d):  # Loop through the Z-stack
        if i == 0:
            ret_img = img.copy()[i][:, :]  # Return image is set to the first slice fo the image
        else:
            temp_img = img.copy()[i][:, :]  # Temporarily store the intensity value of an array
            ret_img[np.where(temp_img > ret_img)] = temp_img[np.where(temp_img > ret_img)]  # Compare the intensity value at each entry, set to the entry where the value is higher

    return ret_img


def enter_video_info(video_path):
    ''' Read video and obtain the inforation of the video
    Args:
        video_path: video_path: (str) path to a single video
    Return:
        df: (pandas.DataFrame) information of the current video
    '''
    video = imread(video_path)
    num_frame, height, width = video.shape
    min_intensity, max_intensity = video.min(), video.max()

    num_channel = get_num_channel(video_path)
    video_name, channel = get_video_name_and_channel(video_path, num_channel)

    df = pd.DataFrame({'video_name':video_name, 'num_frame': num_frame, 'num_channel': num_channel,
                        'channel': channel, 'width': width, 'height': height, 'video_path': video_path,
                        'max_intensity': max_intensity, 'min_intensity': min_intensity, 
                        'model_intensity_all':'linear_plateau', 'init_intensity_all':'0.6_0.8_1', 'analyze':'yes', 'avi_output':'general',}, index=[0])
    return df


def get_num_channel(video_path):
    ''' Find the number of channel of the previously processed video
    Args:
        video_path: video_path: (str) path to a single video
    Return:
        1 or 2: (int) number of channel
    '''
    root_dir = '/'.join(video_path.split('/')[:-1])
    channel_dict = {'C1':0, 'C2':0}
    for f in os.listdir(root_dir):
        if 'C1' in f: channel_dict['C1'] += 1
        if 'C2' in f: channel_dict['C2'] += 1
    if (channel_dict['C1'] == 1) and (channel_dict['C2'] == 1): return 2
    
    return 1


def get_video_name_and_channel(video_path, num_channel=1):
    ''' Find out the video name alias and the channel of current video
    Args:
        video_path: (str) path to a single video
        num_channel: (int) number of channel
    Return:
        video_name: (str) the name of video
        channel: (int) the channel of the video
    '''

    video_name = video_path.split('/')[-1]
    channel = 0

    if num_channel == 1: return video_name, channel

    if 'C1' in video_name:
        channel = 0
        video_name = '_'.join(video_name.split('C1'))
    if 'C2' in video_name:
        channel = 1
        video_name = '_'.join(video_name.split('C2'))
    return video_name, channel