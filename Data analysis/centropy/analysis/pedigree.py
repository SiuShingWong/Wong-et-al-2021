import os
from tqdm import tqdm

import numpy as np
import pandas as pd

def batch_pedigree(input_dir, output_dir=None, main_channel=0, max_pairing_distance=50, pairing_frames=3):
    ''' Read videos.csv and centrosomes.csv to map out the relationship between centrosomes 
    Args:
        input_dir: (str) directory containing videos.csv and all videos listed in videos.csv
        output_dir: (str) directory to save centrosomes.csv and pedigree.csv
        main_channel: (int) the channel to annotate the old mother / new mother
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
        pairing_frames: (int) The break between consecutive frames to be considered as a single division cluster
    '''
    if not output_dir: output_dir = input_dir
        
    try: dataframe_videos = pd.read_csv(os.path.join(input_dir, 'videos.csv')) 
    except: return "could not find videos.csv"
    try: dataframe_centrosomes = pd.read_csv(os.path.join(input_dir, 'centrosomes.csv')) 
    except: return "could not find centrosomes.csv"
        
    dataframe_videos = dataframe_videos[dataframe_videos['analyze']=='yes']
    dataframe_videos = dataframe_videos[dataframe_videos['channel'] == main_channel]
    
    dataframe_pedigree = pd.DataFrame()
    
    for _, row in tqdm(dataframe_videos.iterrows()):
        video_name = row['video_name']  # Video name from the video dataframe
        try:
            pedigree, dataframe_centrosomes = make_pedigree(dataframe_centrosomes, video_name, max_pairing_distance, pairing_frames, main_channel, False)
            dataframe_pedigree = pd.concat((dataframe_pedigree, pedigree))
        except: print(f'cannot extract pedigree from {video_name}'); continue
            
    dataframe_pedigree.to_csv(os.path.join(output_dir, 'pedigree.csv'), index=False)
    dataframe_centrosomes.to_csv(os.path.join(output_dir, 'centrosomes.csv'), index=False)


def make_pedigree(dataframe_centrosomes, video_name=None, max_pairing_distance=20, pairing_frames=3, main_channel=0, debug=False):
    ''' Create pedigree.csv from centrosomes.csv of a ** single video **. It should be pivoted at one of the chennel
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of a single video (i.e. contain only a unique video_name)
        video_name: (str) video name of this pedigree
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
        pairing_frames: (int) The break between consecutive frames to be considered as a single division cluster
        main_channel: (int) The channel to be used to annotate old mother and new mother
        debug: (bool) print some debug message to the prompt
    Return:
        pedigree.csv of a single video
        centrosomes.csv
    '''
    
    df = dataframe_centrosomes[(dataframe_centrosomes['video_name']==video_name) & (dataframe_centrosomes['channel']==main_channel)] # Filter centrosomes.csv to contain only one channel and video
    original_pedigree = initialize_pedigree(df, max_pairing_distance) # Initialize a pedigree table
    new_pedigree = original_pedigree.copy() # Make a copy of pedigree which will be updated for each iteration
    new_centrosomes = dataframe_centrosomes.copy() # Likewise
    
    if debug and check_pairs(original_pedigree, df, max_pairing_distance): return 'initial debugging'
    
    for i, row in original_pedigree.iloc[::-1].iterrows(): # descending order from later frame to earlier frame (From leaf to root)
        if row.isnull().any(): continue
        
        om_id, nm_id = row['parents'], row['particle']
        frame = row['split_frame']
        
        om_intensity = df[(df['particle']==om_id) & (df['frame']==frame)]['total_intensity'].values[0]
        nm_intensity = df[(df['particle']==nm_id) & (df['frame']==frame)]['total_intensity'].values[0]
        
        # If new mother intensity > old mother intensity, update particle id
        if nm_intensity > om_intensity:
            # swao the old mother and new mother annotation on pedigree.csv and centrosomes.csv
            new_pedigree, cache = swap_om_nm_pedigree(new_pedigree, i)
            new_centrosomes = swap_om_nm_centrosomes(new_centrosomes, video_name, frame, cache)
            
    # Remove the annotation if fail to fulfill certain constraints
    new_pedigree, cache = trim_pedigree(new_pedigree, pairing_frames)
    new_centrosomes = trim_centrosomes(new_centrosomes, video_name, cache)
    new_centrosomes = annotate_om_nm(new_pedigree, new_centrosomes, video_name)
    
    # Check the new label still fit under the distance constraint
    df_check = new_centrosomes[(new_centrosomes['video_name']==video_name) & (new_centrosomes['channel']==main_channel)]

    if debug and check_pairs(new_pedigree, df_check, max_pairing_distance): return 'final debugging'
    
    # Add the video name to pedigree
    new_pedigree['video_name'] = video_name

    return new_pedigree, new_centrosomes


def initialize_pedigree(dataframe_centrosomes, max_pairing_distance=20):
    ''' Initialize pedigree.csv from centrosomes.csv of a **single videos** and a **single channel** 
        by the finding the starting frames of all tracks and pairing them
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of a **single video** of a **single channel**
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
    '''
    pairs_at_frames = pairing_at_frames(dataframe_centrosomes, max_pairing_distance) # dictionary of frame and separating particle
    df = pd.DataFrame(columns=['particle', 'parents', 'split_frame'])
    cache = []
    for frame in pairs_at_frames.keys(): # Loop through each frame (Ascending order)
        for pair_0, pair_1 in pairs_at_frames[frame]: # Loop through pairs in each frame
            if pair_0 not in cache:
                cache.append(pair_0)
                if pair_1 not in cache:
                    cache.append(pair_1)
                    # randomly assign one of them to be the parents
                    df = df.append({'particle':pair_0, 'split_frame':frame}, ignore_index=True)
                    df = df.append({'particle':pair_1, 'parents':pair_0, 'split_frame':frame},ignore_index=True)
                else:
                    # pair 1 will be the parents of pair 0
                    df = df.append({'particle':pair_0, 'parents':pair_1, 'split_frame':frame},ignore_index=True)
            else:
                if pair_1 not in cache:
                    # pair 0 will be the parents of pair 1
                    cache.append(pair_1)
                    df = df.append({'particle':pair_1, 'parents':pair_0, 'split_frame':frame},ignore_index=True)
    return df


def pairing_at_frames(dataframe_centrosomes, max_pairing_distance=20):
    ''' Take the centrosomes.csv of a single video and return a dictionary of the following format:
    {f0:[[p0, p1], [p3, p4], ...], f1:[...]} where f is frame number; p is the particle id
    It finds separating pairs of centrosomes under distance threshold at different frames
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of a **single video** of a **single channel**
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
    Return:
        (dict) of {f0:[[a0, b0], [a1, b1], ...], f1:[...], ...}
            f0: (int) the frame of centrosome separation
            a0, b0: (int) or (float) particle id of the pair
    '''
    cache = [] # Checking if we have seen the particle id
    separation_frames, _ = potential_separation_frames(dataframe_centrosomes) # Estimate the potential frame when centrosome splits
    
    pairs_at_frames = dict()
    for frame in separation_frames:
        
        centrosomes_at_frame = dataframe_centrosomes[dataframe_centrosomes['frame']==frame] # data at this frame
        index2particle = {x: y for x, y in enumerate(centrosomes_at_frame['particle'].unique())}
        locations = list(zip(centrosomes_at_frame['x'], centrosomes_at_frame['y'])) # A list of coordinates of centrosomes
        pairs = pair_points_dist(locations, max_pairing_distance) # Get the pair in natural number index
        
        if type(pairs) != np.ndarray: continue
        if pairs.size < 1: continue
        
        pairs = reduce_uniq_id_matches(pairs)
        # Mapping pairs to their original particle id; orders according to ascending frames
        pairs = np.vectorize(lambda i: index2particle[i] if i in index2particle.keys() else np.nan)(pairs)
        
        # For deciding which pairs to add into the dictionary
        # Some pairs stay close for several frames, I only picked the first appearance of a pair
        temp = []
        for pair in pairs: 
            if pair[0] in cache and pair[1] in cache: continue
            else:
                temp.append([pair[0], pair[1]])
                if pair[0] not in cache: cache.append(pair[0])
                if pair[1] not in cache: cache.append(pair[1])
        if temp: pairs_at_frames[frame] = temp
            
    return pairs_at_frames


def potential_separation_frames(dataframe_centrosomes):
    ''' Fetch the start of each track
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of a **single video** of a **single channel**
    return:
        (numpy.array) unique frame number of separation event
        (list) frames of all separation events
    '''
    frames = []
    particles = dataframe_centrosomes['particle'].unique()
    for particle in particles:
        frames.append(dataframe_centrosomes[dataframe_centrosomes['particle']==particle]['frame'].min())
    return np.unique(frames), frames


def pair_points_dist(points, thresh=50):
    '''
    Find all pairs by (x, y) coordinates of every point
    '''
    from sklearn.metrics.pairwise import pairwise_distances
    from scipy.optimize import linear_sum_assignment

    dist_matrix = pairwise_distances(points)  # point: (x,y)
    # set diags to be super high.
    dist_matrix += 100000 * np.diag(np.ones(len(dist_matrix)))
    id1, id2 = linear_sum_assignment(dist_matrix)
    # for each id we iterate and check the distances.
    valid_pairs = []
    for i in range(len(id1)):
        dist12 = dist_matrix[id1[i], id2[i]]
        if dist12 <= thresh:
            valid_pairs.append([id1[i], id2[i]])
    if len(valid_pairs) > 0:
        valid_pairs = np.vstack(valid_pairs)
    return valid_pairs


def reduce_uniq_id_matches(matches_array):
    uniq_matches = []
    for match in matches_array:
        if arreq_in_list(match, uniq_matches) or arreq_in_list(match[::-1], uniq_matches):
            continue
        else:
            uniq_matches.append(match)
    return np.vstack(uniq_matches)


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def check_pairs(pedigree, dataframe_centrosomes, max_pairing_distance=20):
    ''' Check every pair in pedigree.csv if they are within the pairing distance threshold
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of a **single video** of a **single channel**
        max_pairing_distance: (int) distance threshold between a pair in pixel unit
    Return:
        (bool) whether the pairing distance is larger than max_pairing_distance
    '''
    from scipy.spatial import distance
    
    pedigree = pedigree.dropna(axis=0) # remove the row that doesn't have pairs
    for _, row in pedigree.iterrows(): # Loop through row in descending order (From leaf to root)
#         if row.isnull().any(): continue
        om_id = row['parents']
        nm_id = row['particle']
        frame = row['split_frame']
        cond_frame = dataframe_centrosomes['frame']==frame
        # The corrdinates of the pair
        ox = dataframe_centrosomes[(dataframe_centrosomes['particle']==om_id) & cond_frame]['x'].values[0]
        oy = dataframe_centrosomes[(dataframe_centrosomes['particle']==om_id) & cond_frame]['y'].values[0]
        nx = dataframe_centrosomes[(dataframe_centrosomes['particle']==nm_id) & cond_frame]['x'].values[0]
        ny = dataframe_centrosomes[(dataframe_centrosomes['particle']==nm_id) & cond_frame]['y'].values[0]
        
        pairing_distance = distance.euclidean((ox, oy), (nx, ny))
        if pairing_distance > max_pairing_distance:
            print (f'The distance between om_id:{om_id}, nm_id:{nm_id} is greater than {max_pairing_distance}')
            return True
    return False


def swap_om_nm_pedigree(pedigree, row_index):
    ''' Swap the particle id between 'parents' and 'particle' to indicate a change in their relationship
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv of a single video
        row_index: the current row in the pedigree
    ReturnL
        (pandas.DataFrame) updated pedigree
        (tuple) of (int) particle id of updated centrosomes
    '''
    # Get the current index
    cond_this = (pedigree.index==row_index)
    
    # Fetch the updated id from new pedigree
    om_id = pedigree[cond_this]['parents'].values[0]
    nm_id = pedigree[cond_this]['particle'].values[0]
    frame = pedigree[cond_this]['split_frame'].values[0]
    
    # Get when particle id of parents elsewhere before this frame
    cond_particle = ((pedigree['particle']==om_id) & (pedigree['split_frame']<=frame))
    cond_parents = ((pedigree['parents']==om_id) & (pedigree['split_frame']<=frame))
    
    # Update the particle id in the pedigree
    pedigree.loc[cond_this, 'parents'] = nm_id
    pedigree.loc[cond_this, 'particle'] = om_id
    pedigree.loc[cond_particle, 'particle'] = nm_id
    pedigree.loc[cond_parents, 'parents'] = nm_id
    
    return pedigree, (om_id, nm_id)


def swap_om_nm_centrosomes(dataframe_centrosomes, video_name=None, frame=None, pairs=None):
    ''' Update centrosomes.csv relfecting changes of pedigree.csv
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of all centrosomes of all videos
        video_name: (str) video name of current video
        frame: (int) particle will change in frames less less than or equal to this frame
        pairs: (tuple) of (p0, p1) which p0 is former old mother id, p1 is former new mother id
    Return:
        updated centrosomes.csv
    '''
    
    from_particle, to_particle = pairs # pairs to exchange for their ids
    cond_video = (dataframe_centrosomes['video_name']==video_name)
    cond_from_particle = ((dataframe_centrosomes['particle']==from_particle) & 
                          (dataframe_centrosomes['frame']<frame)) # Modifed to centrosome that appeared longer than itself
    cond_to_particle = ((dataframe_centrosomes['particle']==to_particle) & 
                        (dataframe_centrosomes['frame']<frame))

    dataframe_centrosomes.loc[(cond_video & cond_from_particle), 'particle'] = to_particle
    dataframe_centrosomes.loc[(cond_video & cond_to_particle), 'particle'] = from_particle

    return dataframe_centrosomes


def trim_pedigree(pedigree, pairing_frames=3):
    ''' Remove particle from pedigree if it fails constraints: 1) Multigeneration constraint 2) Sibling constraint
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv of a single video
        pairing_frames: (int) The break between consecutive frames to be considered as a single division cluster
    Return:
        (pandas.DataFrame) updated pedigree
        (dict) of {p0:f0, p1:f1, ...}
            p0: (int) particle id
            f0: (int) frame when after this frame, centrosomes should be removed
    '''
    frames = pedigree['split_frame'].unique()
    cluster_dict, num_cluster = find_splitting_cluster(frames, pairing_frames)
    pedigree['split_cluster'] = pedigree['split_frame'].apply(lambda f: cluster_dict[f])
    cache = {}
    for i in range(num_cluster):
        # df_temp = data_pedigree[data_pedigree['split_cluster']==i]
        pedigree, cache = sibling_constraint(pedigree, i, cache)
        pedigree, cache = multigeneration_constraint(pedigree, i, cache)
        
    return pedigree, cache


def find_splitting_cluster(frames, pairing_frames=3):
    ''' Classify continuous separating events as a single division clusters
    Args:
        frames: (list of int) frames where pairs are found
        pairing_frames: (int) The break between consecutive frames to be considered as a single division cluster
    Return:
        (dict) of {f0:c0, f1:c1, ...}
            f: (int) frame of separation
            c: (int) division cluster
        (int) number of separation cluster
    '''
    boundary = np.where(np.diff(frames) > pairing_frames)[0]
    split_cluster = dict()
    curr=0
    # Getting different cluster of frames
    for i in range(len(boundary)+1):
        if i < len(boundary):
            temp = frames[curr:boundary[i]+1]
            curr = boundary[i] + 1
        else: temp = frames[curr:]
        for frame in temp: split_cluster[frame] = i
        num_cluster = i + 1
    return split_cluster, num_cluster


def multigeneration_constraint(pedigree, cluster=0, cache=None):
    '''Constraint to exlude multiple generation (current 3 generations) of family of centrosomes within the single splitting cluster
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv of a single video
        cluster: (int) current splitting cluster
        cache: (dict) of {p0:f0, p1:f1, ...} reference to eliminate pedigree
    Return:
        (pandas.DataFrame) updated pedigree
        (dict) updated cache
    '''
    df = pedigree[pedigree['split_cluster']==cluster]
    # Finding grandmother, old mother, and new mother in the same splitting cluster
    for p in df['particle'].unique():
        pts = df[df['particle']==p]['parents'].values[0] # parents of this particle
        if not df[df['particle']==pts]['parents'].dropna().empty:
            gpts = df[df['particle']==pts]['parents'].values[0] # parents of parents of this particle
            if not df[df['particle']==gpts].dropna(axis=1).empty: # Grandparents in the same spliting cluster
                cache[gpts] = df[df['particle']==gpts]['split_frame'].values[0] # Caching the frame of which the exceptions occur
                cache[pts] = df[df['particle']==pts]['split_frame'].values[0]
                cache[p] = df[df['particle']==p]['split_frame'].values[0]
                to_be_eliminated = pedigree[((pedigree['particle']==gpts) & (pedigree['split_frame']>=cache[gpts])) | 
                                            ((pedigree['parents']==gpts)  & (pedigree['split_frame']>=cache[pts])) | 
                                            ((pedigree['parents']==pts)   & (pedigree['split_frame']>=cache[p]))].index
                pedigree.drop(to_be_eliminated, inplace=True)
    return pedigree, cache
    

def sibling_constraint(pedigree, cluster=0, cache=None):
    '''Constraint to exclude siblings of centrosomes within the single splitting cluster
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv of a single video
        cluster: (int) current splitting cluster
        cache: (dict) of {p0:f0, p1:f1, ...} reference to eliminate pedigree
    Return:
        (pandas.DataFrame) updated pedigree
        (dict) updated cache
    '''
    df = pedigree[pedigree['split_cluster']==cluster]
    if df['parents'].value_counts()[df['parents'].value_counts() > 1].any():
        for pts in df['parents'].value_counts()[df['parents'].value_counts() > 1].index:
            cache[pts] = df[df['parents']==pts]['split_frame'].values[0] # Caching the frame of which the exceptions occur
            to_be_eliminated = pedigree[(pedigree['parents']==pts) & (pedigree['split_frame']>=cache[pts])].index
            pedigree.drop(to_be_eliminated, inplace=True)
    return pedigree, cache


def trim_centrosomes(dataframe_centrosomes, video_name=None, cache=None):
    ''' Update centrosomes.csv relfecting the removal of centrosomes that fail to satisfy constraints
    Args:
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv
        video_name: (str) video_name of current video
        cache: (dict) of {p0:f0, p1:f1, ...}
            p: (int) particle id
            f: (int) frame when after this frame, centrosomes should be removed
    Return:
        updated centrosomes.csv
    '''

    cond_video = (dataframe_centrosomes['video_name']==video_name)
    
    # Remove centrosome as in the exception of pedigree
    for particle in cache.keys():
        frame = cache[particle]
        cond_particle = ((dataframe_centrosomes['particle']==particle) & (dataframe_centrosomes['frame']>=frame))
        # to_be_eliminated = dataframe_centrosomes[cond_video & cond_particle].index
        # dataframe_centrosomes.drop(to_be_eliminated, inplace=True)
        dataframe_centrosomes.loc[(cond_video & cond_particle), 'pair_id'] = np.nan
        dataframe_centrosomes.loc[(cond_video & cond_particle), 'age_type'] = np.nan

    return dataframe_centrosomes


def annotate_om_nm(pedigree, dataframe_centrosomes, video_name=None):
    '''Annotate whether in a pair which one is old mother or new mother and Assign a pair an uniquie id in centrosomes.csv
    Args:
        pedigree: (pandas.DataFrame) pedigree.csv of a single video
        dataframe_centrosomes: (pandas.DataFrame) centrosomes.csv of all centrosomes of all videos
        video_name: (str) video_name of current video
    Return:
        updated centrosomes.csv
    '''
    cond_video = dataframe_centrosomes['video_name']==video_name
    for i, row in pedigree.iterrows():
        
        if row.isnull().any(): continue
            
        om_id, nm_id = row['parents'], row['particle']
        frame = row['split_frame']
        cond_frame = dataframe_centrosomes['frame'] >= frame

        om_cond = (cond_video & cond_frame & (dataframe_centrosomes['particle']==om_id))
        nm_cond = (cond_video & cond_frame & (dataframe_centrosomes['particle']==nm_id))

        dataframe_centrosomes.loc[om_cond, 'age_type'] = 'old_mother'
        dataframe_centrosomes.loc[nm_cond, 'age_type'] = 'new_mother'
        dataframe_centrosomes.loc[om_cond | nm_cond, 'pair_id'] = i
    
    return dataframe_centrosomes