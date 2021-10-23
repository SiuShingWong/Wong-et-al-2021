import os
import pandas as pd

def construct_folders(input_dir):
    fig_dir = os.path.join(input_dir, 'Figures')
    if not os.path.exists(fig_dir): os.mkdir(fig_dir)
        
    individual_dir = os.path.join(fig_dir, 'Individual')
    if not os.path.exists(individual_dir): os.mkdir(individual_dir)

    stat_dir = os.path.join(fig_dir, 'Statistics')
    if not os.path.exists(stat_dir): os.mkdir(stat_dir)

    prism_dir = os.path.join(fig_dir, 'Prism')
    if not os.path.exists(prism_dir): os.mkdir(prism_dir)

    categorical_dir = os.path.join(fig_dir, 'Categorical')
    if not os.path.exists(categorical_dir): os.mkdir(categorical_dir)
    
    correlation_dir = os.path.join(fig_dir, 'Correlation')
    if not os.path.exists(correlation_dir): os.mkdir(correlation_dir)

    collective_dir = os.path.join(fig_dir, 'Collective')
    if not os.path.exists(collective_dir): os.mkdir(collective_dir)

    return {'Figure':fig_dir, 'Individual':individual_dir, 'Statistics':stat_dir, 'Prism':prism_dir, 'Categorical':categorical_dir,
    'Correlation':correlation_dir, 'Collective':collective_dir}

def read_data(input_dir, get_videos=False, get_centrosomes=False, get_manual=False, get_nuclei=False, get_radial=False, get_simulations=False, get_models=False, get_pedigree=False):

    dataframe_videos = None
    dataframe_centrosomes = None
    dataframe_manual = None
    dataframe_nuclei = None
    dataframe_radial = None
    dataframe_simulations = None
    dataframe_models = None
    dataframe_pedigree = None 

    if get_videos:
        video_path = os.path.join(input_dir, 'videos.csv')
        if os.path.exists(video_path):
            dataframe_videos = pd.read_csv(video_path)
            dataframe_videos = dataframe_videos[dataframe_videos['analyze']=='yes'] 
        else:
            print ("could find videos.csv")

    # Dataframe of centrosome tracks and parameters
    if get_centrosomes:
        centrosome_path = os.path.join(input_dir, 'centrosomes.csv')        
        if os.path.exists(centrosome_path):
            dataframe_centrosomes = pd.read_csv(centrosome_path)
        else:
            print ("could find centrosomes.csv")
    
    if get_manual:
        manual_path = os.path.join(input_dir, 'manual.csv')
        if os.path.exists(manual_path):
            dataframe_manual = pd.read_csv(manual_path)
        else:
            print ("could find centrosomes.csv") 
    
    if get_nuclei:
        # Dataframe of nucleus tracks and parameters
        nucleus_path = os.path.join(input_dir, 'nuclei.csv')
        if os.path.exists(nucleus_path):
            dataframe_nuclei = pd.read_csv(nucleus_path)
        else:
            print ("could find nuclei.csv")

    if get_radial:
        # Dataframe of radial profile
        radial_path = os.path.join(input_dir, 'radial.csv')
        if os.path.exists(radial_path):
            dataframe_radial = pd.read_csv(radial_path)
        else:
            print ("could find radial.csv")

    if get_simulations:
        # Dataframe of simulation of curve fitting
        simulation_path = os.path.join(input_dir, 'simulations.csv')
        if os.path.exists(simulation_path):
            dataframe_simulations = pd.read_csv(simulation_path)
        else:
            print ("could find simulations.csv")

    if get_models:
        # Dataframe of model parameters curve fitting
        model_path = os.path.join(input_dir, 'models.csv')
        if os.path.exists(model_path):
            dataframe_models = pd.read_csv(model_path)
        else:
            print ("could find models.csv")

    if get_pedigree:
        # Dataframe of model parameters curve fitting
        pedigree_path = os.path.join(input_dir, 'pedigree.csv')
        if os.path.exists(pedigree_path):
            dataframe_pedigree = pd.read_csv(pedigree_path)
        else:
            print ("could find pedigree.csv")
    
    return {'videos':dataframe_videos, 'centrosomes':dataframe_centrosomes, 'manual':dataframe_manual, 'nuclei':dataframe_nuclei,
    'radial':dataframe_radial, 'simulations':dataframe_simulations, 'models':dataframe_models, 'pedigree':dataframe_pedigree}


def xls2csv(input_dir, stem):
    from pathlib import Path
    for root, _, files in os.walk(input_dir):
        for f in files:
            if stem in files:
                xls = pd.read_csv(os.path.join(root, f), sep='\t')
                xls.to_csv(os.path.join(root, stem + '.csv'), index=False)