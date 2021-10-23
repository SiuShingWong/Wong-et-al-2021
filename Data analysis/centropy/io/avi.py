import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
from skimage.io import imread

# =====================================================================================================
# ================================== batch generation of avi ==========================================
# =====================================================================================================

def batch_general_avi(input_dir, label_centrosome=False, fps=10, clip_auto=[0.05, 0.05], color='green', reuse_contrast=None, 
                      rotation=False, window_size=28, output_dims='auto', label_age=False):
    ''' API that loops through all data in a dataset to generate example movies
    Args:
        input_dir: (str) the location that contains videos.csv and dataset
        label_centrosome: (bool) whether to label centrosomes in the movie
        fps: (int) frame rate
        clip_auto: (list) of [(float), (float)] where both (float) are the percentage of pixel to set as maximum value in channel 0 and channel 1
        color: (str) what color to display if there're more than 1 channel. This argument is ignored
        rotation: (bool) whether to rotate 90o
    '''

    if os.path.exists(os.path.join(input_dir, 'videos.csv')):                   # Check if videos.csv is in input_dir
        dataframe_videos = pd.read_csv(os.path.join(input_dir, 'videos.csv'))
        dataframe_videos = dataframe_videos[dataframe_videos['avi_output'].notnull()]  # videos.csv requires a column named 'avi' to signal the generation of example movies
    else: return ('No videos.csv found in {}'.format(input_dir)) # videos.csv is required to be in the directory

    if label_centrosome:
        if os.path.exists(os.path.join(input_dir, 'centrosomes.csv')): # Check if centrosomes.csv or manaual_centrosomes.csv is in input_dir
            dataframe_centrosomes = pd.read_csv(os.path.join(input_dir, 'centrosomes.csv'))
        elif os.path.exists(os.path.join(input_dir, 'manual.csv')):
            dataframe_centrosomes = pd.read_csv(os.path.join(input_dir, 'manual_centrosomes.csv'))
        else: dataframe_centrosomes = None
    else: dataframe_centrosomes = None
    
    for _, row in tqdm(dataframe_videos.iterrows()): # Loop through all video
        video_name = row['video_name']
        video_path = row['video_path']
        mode = row['avi_output'] # Check the mode of movie generation
        
        if not os.path.exists(video_path): continue

        root_dir = '/'.join(video_path.split('/')[:-1]) # creating directory for storing example movies
        output_dir = os.path.join(root_dir, 'label')
        if not os.path.exists(output_dir): os.mkdir(output_dir)

        if mode=='general': # general mode of example movie generation
            general_avi(dataframe_videos, dataframe_centrosomes, dataframe_nuclei, video_name, output_dir, label_centrosome=label_centrosome, 
                    fps=fps, clip_auto=clip_auto, reuse_contrast=reuse_contrast, color=color, rotation=rotation, label_age=label_age)

# =====================================================================================================
# =====================================================================================================
# =====================================================================================================



# =====================================================================================================
# ================================== different types of avi movies ====================================
# =====================================================================================================
def general_avi(dataframe_videos, dataframe_centrosomes, video_name, output_dir, label_centrosome=False, 
                fps=15, clip_auto=[0.05, 0.05], color='green', reuse_contrast=None, rotation=False, label_age=False):

    ''' Generate typical microscopy movie in avi format for display
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv or manual_centrosomes.csv
        dataframe_nuclei: (pandas.DataFrame) nuclei.csv
        video_name: (str) video name
        output_dir: (str) the location to store the output movies
        label_centrosome: (bool) whether to label centrosomes in the movie
        fps: (int) frame rate
        clip_auto: (list) of [(float), (float)] where both (float) are the percentage of pixel to set as maximum value in channel 0 and channel 1
        color: (str) what color to display if there're more than 1 channel. This argument is ignored
        rotation: (bool) whether to rotate 90o
    '''
    ############# Reading data and deciding setting #############
    this_df_videos = dataframe_videos[(dataframe_videos['video_name']==video_name)&(dataframe_videos['channel']==0)]
    
    if dataframe_centrosomes is not None: this_df_centrosomes = dataframe_centrosomes[(dataframe_centrosomes['video_name']==video_name)&(dataframe_centrosomes['channel']==0)]
    if dataframe_nuclei is not None: this_df_nucleus = dataframe_nuclei[dataframe_nuclei['video_name']==video_name]
    #############################################################

    ################# Initialize example movie ##################
    temp_video, (height, width, nframes, nchannels) = read_and_adjust_video(dataframe_videos, video_name, clip_auto, color, reuse_contrast)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    if rotation: (width, height) = (height, width)
    output_video = cv2.VideoWriter(os.path.join(output_dir, f'{video_name}_general_label-{label_centrosome}_label-age-{label_age}.avi'), fourcc, fps, (width, height))
    #############################################################

    ################### Output example movie ####################
    for i in range(nframes): # Loop through every frame
        
        img = color_rotation_frame(temp_video, i, color, rotation) # adjust the current frame from video       

        if label_centrosome:
            if dataframe_centrosomes is not None: # label centrosome
                this_df_centrosomes_i = this_df_centrosomes[this_df_centrosomes['frame']==i]
                if not (this_df_centrosomes_i.empty):
                    img = label_centrosomes(img, this_df_centrosomes_i, rotation, label_age)
        
        output_video.write(img)
    
    cv2.destroyAllWindows()
    output_video.release()
    #############################################################


# ==============================================================================================================
# =============================== helper function for generation of avi movies =================================
# ==============================================================================================================

def label_centrosomes(img, dataframe_centrosomes_i, rotation, label_age=False):
    ''' label centrosome on the image
    Args:
        img: (np.array) of shape (height, width, 3) BGR image
        dataframe_centrosomes_i: (pandas.DataFrame) centrosomes.csv at certain frame
        rotation: (bool) whether to rotate 90o
    '''
    height, width, _ = img.shape
    for _, row in dataframe_centrosomes_i.iterrows(): # loop through all centrosomes in the current frame
        p = row['particle']
        x, y = row['x'], row['y']
        if rotation:
            x, y = width - y, x
        pair_id = row['pair_id']
        if label_age:
            if not np.isnan(pair_id):
                if row['age_type']=='old_mother': thickness=2
                else: thickness=1
                cv2.circle(img, (int(x), int(y)), 12, (0, 0, 255), thickness)
                cv2.putText(img, str(pair_id), (int(x)-10, int(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), 1)
            cv2.putText(img, str(p), (int(x)-10, int(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def label_nuclei(img, dataframe_nuclei_i, rotation):
    ''' label nuclei on the image
    Args:
        img: (np.array) of shape (height, width, 3) BGR image
        dataframe_nuclei_i: (pandas.DataFrame) nuclei.csv at certain frame
        rotation: (bool) whether to rotate 90o
    '''
    height, width, _ = img.shape
    for _, row in dataframe_nuclei_i.iterrows(): # loop through all nuclei in the current frame
        p = row['particle']
        x, y = row['x'], row['y']
        r = row['radius']
        if rotation:
            x, y = width - y, x
        cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
    return img


def read_and_adjust_video(dataframe_videos, video_name, clip_auto, color, reuse_contrast=None):
    ''' adjust the video color and brightness
    Args:
        dataframe_videos: (pandas.DataFrame) videos.csv
        video_name: (str) video name
        clip_auto: (list) of [(float), (float)] where both (float) are the percentage of pixel to set as maximum value in channel 0 and channel 1
        color: (str) what color to display if there're more than 1 channel. This argument is ignored
    '''
    this_df_videos = dataframe_videos[(dataframe_videos['video_name']==video_name)&(dataframe_videos['channel']==0)] # filter the information of this video
    height, width, nframes, nchannels = int(this_df_videos['height'].values[0]), int(this_df_videos['width'].values[0]), int(this_df_videos['num_frame'].values[0]), int(this_df_videos['num_channel'].values[0])
    temp_video = np.zeros((nframes, height, width, 3), dtype=np.uint8)
                    
    for c in range(nchannels): # Loop through channels
        this_df_videos = dataframe_videos[(dataframe_videos['video_name']==video_name)&(dataframe_videos['channel']==c)]
        video_path = this_df_videos['video_path'].values[0]
        video = imread(video_path)

        _, alpha, beta = auto_contrast_brightness(video[0,:,:], clip_hist_percent=clip_auto[c]) # Detect the brightness in current frame
        adjusted_video = cv2.convertScaleAbs(video, alpha=alpha, beta=beta) # apply the adjustment to the whole video
        
        if nchannels==1:
            if color=='green':
                temp_video[:,:,:,1] = adjusted_video
            elif color=='red':
                temp_video[:,:,:,2] = adjusted_video
            elif color=='blue':
                temp_video[:,:,:,0] = adjusted_video
            elif color=='gray':
                return adjusted_video, (height, width, nframes, nchannels)
        else:
            temp_video[:,:,:,2-c] = adjusted_video
    
    return temp_video, (height, width, nframes, nchannels)


def color_rotation_frame(video, frame, color, rotation):
    ''' Convert the current frame into RGB fomrat and rotate properly
    Args:
        video: (np.array) of shape (frame, height, width) or (frame, height, width, 3)
        frame: (int) the current frame
        color: (str) can be 'gray'
        rotation: (bool) whether to rotate 90o
    '''
    if color == 'gray':
        img = video[frame,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = video[frame,:,:,:]
    if rotation:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================



# ==============================================================================================================
# ===================================== image adjustment for visualization =====================================
# ==============================================================================================================
def auto_contrast_brightness(img, clip_hist_percent=0.5):
    ''' Adopted from https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
    It will adjust the contrast or brightness by saturating a small percentage of pixel
    '''
    import cv2
    # Calculate grayscale histogram
    hist_max = np.iinfo(img.dtype).max + 1

    hist = cv2.calcHist([img], [0], None, [hist_max], [0,hist_max])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    # alpha = 255 / (maximum_gray - minimum_gray) + 0.5
    beta = -minimum_gray * alpha
    # beta = -( minimum_gray - 10) * alpha
    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # print (alpha, beta)
    return auto_result, alpha, beta


def contrast_brightness(img,alpha,beta):
    img = img * alpha + beta
    img = img.astype(int)

    img[img > 255] = 255

    img = np.uint8(img)

    return img


def minmax_contrast_brightness(img):
    # Change the function name from auto --> min_max
    max_val = np.amax(img)
    min_val = np.amin(img)

    max_val = int(max_val)
    min_val = int(min_val)

    alpha = 255 / (max_val - min_val)
    beta = -min_val * alpha

    img = contrast_brightness(img, alpha, beta)

    print('minmax')
    print(img)
    return img


def adjust_nucleus_image(img, contrast_method='minmax', clip_hist_percent=0.5,protein_type="normal"):
    import cv2

    if protein_type == "normal":
        img = cv2.convertScaleAbs(img)
    
    if contrast_method=='minmax':
        img = minmax_contrast_brightness(img)
    elif contrast_method=='auto':
        img, _, _ = auto_contrast_brightness(img, clip_hist_percent=clip_hist_percent)


    # Turning the image to uint8
    img = np.int16(img)
    img = np.uint8(img)

    return img
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
