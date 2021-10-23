import os
import numpy as np
import pandas as pd
from centropy.statistics.verbrose import get_sig
from centropy.io import helper

def batch_hypothesis_testing(input_dir, output_dir=None, attribute_dict=None, comparison_type=None, between_type=None, res_threshold=0.1, frame_rate=0.5):
    ''' Perform hypothesis testing to compare across comparison type of possible parameters provided by attribute_dict
    Args:
        input_dir: (str) where videos.csv and models.csv are saved
        output_dir: (str) directory to save statistics
        attribute_dict: (dict) {a0:[c0, c1, ...], a1:[...], ...}
            a: (str) e.g. 'area', 'intensity', or 'density'
            c: (str) parameters e.g. 'increase_rate_m', ...
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        between_type: (str) 'cycle' or 'manipulated_protein' or 'age_type'
        res_threshold: (float) threshold to remove outlier
        frame_rate: (float) conversion ratio from frame to time
    '''
    from centropy.analysis.dataframe import filter_merge_videos

    dataframe_dict = helper.read_data(input_dir, get_videos=True, get_models=True)
    dataframe_videos = dataframe_dict['videos']
    dataframe_models = dataframe_dict['models']
    if (dataframe_videos is None) or (dataframe_models is None): return "quit batch_hypothesis_testing"
    if output_dir is None: output_dir = helper.construct_folders(input_dir)['Statistics']
    if attribute_dict is None: attribute_dict = { 'intensity': ['initial_r', 'peak_r', 'added_r','peak_time_r', 'increase_rate_r', 's-phase_duration'],}

    merged_dataframe = filter_merge_videos(dataframe_videos, dataframe_models, res_threshold, frame_rate)
    num_comparison = len(merged_dataframe[comparison_type].unique())
    
    df_infos = pd.DataFrame()
    df_mcs = pd.DataFrame()

    if num_comparison==2:
        for channel in merged_dataframe['channel'].unique():
            for attribute in list(attribute_dict.keys()):
                parameters = attribute_dict[attribute]
                df_info, df_mc = two_group_statistics(merged_dataframe, channel, attribute, parameters, comparison_type)
                df_infos = pd.concat((df_infos, df_info))
                df_mcs = pd.concat((df_mcs, df_mc))
                
    elif num_comparison>2:
        df_one_ways = pd.DataFrame()
        for channel in merged_dataframe['channel'].unique():
            for attribute in list(attribute_dict.keys()):
                parameters = attribute_dict[attribute]
                df_info, df_one_way, df_mc = one_way_anova_statistics(merged_dataframe, channel, attribute, parameters, comparison_type)
                df_infos = pd.concat((df_infos, df_info))
                df_mcs = pd.concat((df_mcs, df_mc))
                df_one_ways = pd.concat((df_one_ways, df_one_way))
        df_one_ways.reset_index(inplace=True)
        df_one_ways.drop(labels='index', axis=1, inplace=True)      
        df_one_ways.to_csv(os.path.join(output_dir, 'one_way_anova.csv'))

    else:
        return "Could not perform comparison. Number of groups is < 2."
                
    df_infos.reset_index(inplace=True)
    df_infos.drop(labels='index', axis=1, inplace=True)
    df_infos.to_csv(os.path.join(output_dir, 'statistics_summary.csv'))
    
    df_mcs.reset_index(inplace=True)
    df_mcs.drop(labels='index', axis=1, inplace=True)        
    df_mcs.to_csv(os.path.join(output_dir, 'multiple_comparison.csv'))
    
    
    if between_type != None:
        df_mcs = pd.DataFrame()
        df_two_ways = pd.DataFrame()
        for channel in merged_dataframe['channel'].unique():
            for attribute in list(attribute_dict.keys()):
                parameters = attribute_dict[attribute]
                df_two_way, df_mc = two_way_anova_statistics(merged_dataframe, channel, attribute, parameters, comparison_type, between_type)
                df_two_ways = pd.concat((df_two_ways, df_two_way))
                df_mcs = pd.concat((df_mcs, df_mc))
                
        df_two_ways.reset_index(inplace=True)
        df_two_ways.drop(labels='index', axis=1, inplace=True)      
        df_two_ways.to_csv(os.path.join(output_dir, f'two_way_anova of {comparison_type} and {between_type}.csv'))
        
        df_mcs.reset_index(inplace=True)
        df_mcs.drop(labels='index', axis=1, inplace=True)        
        df_mcs.to_csv(os.path.join(output_dir, f'two_way_multiple_comparison of {comparison_type} and {between_type}.csv'))


def two_group_statistics(merged_dataframe, channel=None, attribute=None, parameters=None, comparison_type=None):
    '''Determine what statistical test to use between 2 group comparisons
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameters: (list) of parameters e.g. ['increase_rate_m', 'added_m', ...]
        comparison_type: (str) 'cycle' or 'manipulated_protein'
    '''

    from pingouin import normality, homoscedasticity

    df = merged_dataframe[(merged_dataframe['attribute']==attribute) & 
                          (merged_dataframe['age_type']=='all') & 
                          (merged_dataframe['channel']==channel)]
    
    df_mc = pd.DataFrame()
    df_info = pd.DataFrame()

    for parameter in parameters:
        
        try:
            normality_test_name = 'DAgostino and Pearson' # Test for normality
            normality_test = normality(df, dv=parameter, group=comparison_type, method='normaltest', alpha=0.05)
            is_normal = False not in normality_test['normal'].unique()
        except ValueError:
            normality_test_name = 'Sample size < 8'
            is_normal = False
        
        if is_normal:
            variance_test_name = 'Levene'
            variance_test = homoscedasticity(df, dv=parameter, group=comparison_type, method='levene', alpha=0.05)
            variance_is_equal = False not in variance_test['equal_var'].unique()
            anova_name = np.nan
            mc_name = 'unpaired t-test'
            mc_test = two_group_mc_test(df, mc_name, channel, attribute, parameter, comparison_type)
        
        else:
            variance_test_name = np.nan
            variance_is_equal = np.nan
            anova_name = np.nan
            mc_name = 'Mann-Whitney'
            mc_test = two_group_mc_test(df, mc_name, channel, attribute, parameter, comparison_type)
            
        df_info = df_info.append(pd.Series({'channel':channel, 'attribute':attribute, 'parameter':parameter, 'contrast':comparison_type,
                       'normality_test':normality_test_name, 'is_normal':is_normal, 'variance_test': variance_test_name, 'variance_is_equal': variance_is_equal,
                       'ANOVA':anova_name, 'Multiple_Testing':mc_name}), ignore_index=True)
        df_mc = pd.concat((df_mc, mc_test))

    df_mc.reset_index(inplace=True)
    df_mc.drop(labels='index', axis=1, inplace=True)
    
    return df_info, df_mc


def two_group_mc_test(merged_dataframe, mc_name=None, channel=None, attribute=None, parameter=None, comparison_type=None):
    ''' Perform hypothesis testing between 2 groups of comparison
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        mc_name: (str) name of the statistical test used
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameter: (str) parameters e.g. 'increase_rate_m'
        comparison_type: (str) 'cycle' or 'manipulated_protein'
    '''
    from pingouin import pairwise_ttests
    
    col_order = ['channel', 'attribute', 'parameter', 'contrast', 'A', 'B', 'parametric', 'tail', 'statistical_test', 'correction_method', 'p-val', 'significance']

    if mc_name == 'Mann-Whitney':
        df = pairwise_ttests(merged_dataframe, dv=parameter, between=comparison_type, parametric=False, correction='auto')
        df.rename(columns={'Contrast':'contrast', 'Parametric':'parametric', 'Tail':'tail', 'p-unc': 'p-val'}, inplace=True)
        if 'hedges' in df.columns: df.drop(columns=['Paired', 'hedges'], inplace=True)
        else: df.drop(columns=['Paired'], inplace=True)
        df['statistical_test'] = 'Mann-Whitney U'
        df['significance'] = df['p-val'].apply(get_sig)
        
    elif mc_name == 'unpaired t-test':
        df = pairwise_ttests(merged_dataframe, dv=parameter, between=comparison_type, parametric=True, correction='auto')
        df.rename(columns={'Contrast':'contrast', 'Parametric':'parametric', 'Tail':'tail', 'p-unc': 'p-val'}, inplace=True)
        df.drop(columns=['Paired','hedges', 'T', 'dof', 'BF10'], inplace=True)
        df['statistical_test'] = 'unpaired t-test'
        df['significance'] = df['p-val'].apply(get_sig)
        
    else:
        df = pd.DataFrame([{y: np.nan for _, y in enumerate(col_order)}])
        
    df['correction_method'] = np.nan
    df['channel'] = channel
    df['attribute'] = attribute
    df['parameter'] = parameter
    df['contrast'] = comparison_type
    df = df[col_order]
    
    return df


def one_way_anova_statistics(merged_dataframe, channel=None, attribute=None, parameters=None, comparison_type=None):
    ''' One-way ANOVA for groups > 2 and varies dependent on the normality and homoscedaticity of the data
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameters: (list) of parameters e.g. ['increase_rate_m', 'added_m', ...]
        comparison_type: (str) 'cycle' or 'manipulated_protein'
    '''
    from pingouin import kruskal, homoscedasticity, welch_anova, anova, normality

    df = merged_dataframe[(merged_dataframe['attribute']==attribute) & 
                          (merged_dataframe['age_type']=='all') & 
                          (merged_dataframe['channel']==channel)]
    
    df_mc = pd.DataFrame()
    df_anova = pd.DataFrame()
    df_info = pd.DataFrame()
    
    for parameter in parameters:
        try:
            normality_test_name = 'DAgostino and Pearson' # Test for normality
            normality_test = normality(df, dv=parameter, group=comparison_type, method='normaltest', alpha=0.05)
            is_normal = False not in normality_test['normal'].unique()
        except ValueError:
            normality_test_name = 'Sample size < 8'
            is_normal = False
        
        
        if is_normal:
            variance_test_name = 'Levene'
            variance_test = homoscedasticity(df, dv=parameter, group=comparison_type, method='levene', alpha=0.05)
            variance_is_equal = False not in variance_test['equal_var'].unique()
            if variance_is_equal:
                anova_name = 'Ordinary ANOVA'
                anova_test = anova(df, dv=parameter, between=comparison_type, detailed=False)
                anova_pval = anova_test['p-unc'].values[0]
                anova_significance = get_sig(p_val=anova_pval)
                
                if anova_pval < 0.05: # If the ANOVA p-value is significant
                    mc_name = 'Tukey-Kramer'
                    mc_test = one_way_mc_test(df, mc_name, channel, attribute, parameter, comparison_type)
                else:
                    mc_name = np.nan
                    mc_test = one_way_mc_test(df, None, channel, attribute, parameter, comparison_type)
            else:
                anova_name = 'Welch ANOVA'
                anova_test = welch_anova(df, dv=parameter, between=comparison_type)
                anova_pval = anova_test['p-unc'].values[0]
                anova_significance = get_sig(p_val=anova_pval)
                
                if anova_pval < 0.05: # If the ANOVA p-value is significant
                    mc_name = 'Games-Howell'
                    mc_test = one_way_mc_test(df, mc_name, channel, attribute, parameter, comparison_type)
                else:
                    mc_name = np.nan
                    mc_test = one_way_mc_test(df, None, channel, attribute, parameter, comparison_type)
        
        else:
            variance_test_name = np.nan
            variance_is_equal = np.nan
            anova_name = 'Kruskal-Wallis'
            anova_test = kruskal(df, dv=parameter, between=comparison_type) 
            anova_pval = anova_test['p-unc'].values[0]
            anova_significance = get_sig(p_val=anova_pval)
            
            if anova_pval < 0.05: # Perfrom multiple comparison
                mc_name = 'Mann-Whitney U'
                mc_test = one_way_mc_test(df, mc_name, channel, attribute, parameter, comparison_type)
            else: 
                mc_name = np.nan
                mc_test = one_way_mc_test(df, None, channel, attribute, parameter, comparison_type)
                
                
        df_info = df_info.append(pd.Series({'channel':channel, 'attribute':attribute, 'parameters':parameter, 
                    'contrast':comparison_type, 'normality_test': normality_test_name, 'is_normal': is_normal, 
                    'variance_test': variance_test_name, 'variance_equal': variance_is_equal, 'ANOVA':anova_name, 
                    'multiple_testing':mc_name}), ignore_index=True)
        df_anova = df_anova.append(pd.Series({'channel':channel, 'attribute':attribute, 'parameters':parameter, 
                    'contrast':comparison_type, 'anova_test':anova_name, 'p-val': anova_pval, 
                    'significance': anova_significance}), ignore_index=True)
        df_mc = pd.concat((df_mc, mc_test))
        
    df_mc.reset_index(inplace=True)
    df_mc.drop(labels='index', axis=1, inplace=True)
    
    return df_info, df_anova, df_mc


def one_way_mc_test(merged_dataframe, mc_name=None, channel=None, attribute=None, parameter=None, comparison_type=None):
    ''' carry out multiple comparison based on the test name for a particular parameter and attribute and standardize the comparison summary for different test
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        mc_name: (str) name of the statistical test used
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameter: (str) parameters e.g. 'increase_rate_m'
        comparison_type: (str) 'cycle' or 'manipulated_protein'
    '''
    from pingouin import pairwise_ttests, pairwise_gameshowell, pairwise_tukey

    # Pre select data for channel
    col_order = ['channel', 'attribute', 'parameter', 'contrast', 'A', 'B', 'parametric', 'tail', 'statistical_test', 'correction_method', 'p-val', 'significance'] # For ordering the column of dataframe
    
    # It's arranged in 1)Statistical test and 2) Cleaning the dataframe for visibility
    # Note that if user want to inspect specific statistics, they can called the original pg library
    if mc_name == 'Mann-Whitney U':
        df = pairwise_ttests(merged_dataframe, dv=parameter, between=comparison_type, correction=True, parametric=False, padjust='bonf')
        df.rename(columns={'Contrast':'contrast', 'Parametric':'parametric', 'Tail':'tail', 'p-corr': 'p-val', 'p-adjust':'correction_method'}, inplace=True)
        df.drop(columns=['Paired', 'U-val', 'p-unc', 'hedges'], inplace=True)
        df['statistical_test'] = 'Mann-Whitney U'
        df['correction_method'] = 'bonf'
        df['significance'] = df['p-val'].apply(get_sig)
        
    elif mc_name == 'Games-Howell':
        df = pairwise_gameshowell(merged_dataframe, dv=parameter, between=comparison_type)
        df.rename(columns={'pval': 'p-val'}, inplace=True)
        df.drop(columns=['mean(A)', 'mean(B)', 'diff', 'se', 'T', 'hedges', 'df'], inplace=True)
        df['correction_method'] = np.nan
        df['parametric'] = np.nan
        df['statistical_test'] = 'Games-Howell'
        df['significance'] = df['p-val'].apply(get_sig)
        
    elif mc_name == 'Tukey-Kramer':
        df = pairwise_tukey(merged_dataframe, dv=parameter, between=comparison_type)
        df.rename(columns={'p-tukey': 'p-val'}, inplace=True)
        df.drop(columns=['mean(A)', 'mean(B)', 'diff', 'se', 'T', 'hedges'], inplace=True)
        df['correction_method'] = np.nan
        df['parametric'] = np.nan
        df['statistical_test'] = 'Tukey-Kramer'
        df['significance'] = df['p-val'].apply(get_sig)
        
    else:
        df = pd.DataFrame([{y: np.nan for _, y in enumerate(col_order)}])
        
    df['channel'] = channel
    df['attribute'] = attribute
    df['parameter'] = parameter
    df['contrast'] = comparison_type
    df = df[col_order]
    
    return df


def two_way_anova_statistics(merged_dataframe, channel=None, attribute=None, parameters=None, comparison_type=None, between_type='age_type'):
    '''Perform nonparametric 2-way ANOVA across all parameters of a particular attribute.
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameters: (list) of parameters e.g. ['increase_rate_m', 'added_m', ...]
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        between_type: (str) 'cycle' or 'manipulated_protein' or 'age_type'
    '''
    from pingouin import anova

    if between_type=='age_type' or comparison_type=='age_type':
        df = merged_dataframe[(merged_dataframe['attribute']==attribute) & 
                              (merged_dataframe['age_type']!='all') & 
                              (merged_dataframe['channel']==channel)]
    else:
        df = merged_dataframe[(merged_dataframe['attribute']==attribute) & 
                              (merged_dataframe['age_type']=='all') & 
                              (merged_dataframe['channel']==channel)]
    
    df['age_type'].cat.remove_unused_categories(inplace=True)
    df['cycle'] = df['cycle'].astype(str)
    df['manipulated_protein'] = df['manipulated_protein'].astype(str)
    
    df_anova = pd.DataFrame()
    df_mc = pd.DataFrame()

    for parameter in parameters:
        if parameter=='s-phase_duration': continue
        anova_test = anova(df, dv=parameter, between=[comparison_type, between_type], detailed=False) # 2-way ANOVA
        
        anova_test.rename(columns={'Source':'contrast', 'p-unc':'p-val'}, inplace=True)
        anova_test.drop(index=3, inplace=True)
        anova_test['channel'] = channel
        anova_test['attribute'] = attribute
        anova_test['parameter'] = parameter
        anova_test['statistical_test'] = '2-way ANOVA'
        anova_test['significance'] = anova_test['p-val'].apply(get_sig)
        mc_test = two_way_mc_test(df, 'Mann-Whitney MC', channel, attribute, parameter, comparison_type, between_type) # Multiple comparison
        df_anova = pd.concat((df_anova, anova_test))
        df_mc = pd.concat((df_mc, mc_test))
        
    df_mc.reset_index(inplace=True)
    df_mc.drop(labels='index', axis=1, inplace=True)
    df_anova.reset_index(inplace=True)
    df_anova.drop(labels='index', axis=1, inplace=True)
    
    return df_anova, df_mc


def two_way_mc_test(merged_dataframe, mc_name=None, channel=None, attribute=None, parameter=None, comparison_type=None, between_type='age_type'):
    '''Perfrom nonparametric multiple testing after 2-way ANOVA
    Args:
        merged_dataframe: (pandas.DataFrame) merged from videos.csv and models.csv
        mc_name: (str) name of the statistical test used
        channel: (int) the current channel
        attribute: (str) 'area', 'intensity', ...
        parameter: (str) parameters e.g. 'increase_rate_m'
        comparison_type: (str) 'cycle' or 'manipulated_protein'
        between_type: (str) 'cycle' or 'manipulated_protein' or 'age_type'
    '''
    from pingouin import pairwise_ttests

    col_order = ['channel', 'attribute', 'parameter', 'contrast', 'A', 'B', 'parametric', 'tail', 'statistical_test', 'correction_method', 'p-val', 'significance']

    if mc_name == 'Mann-Whitney MC':
        df = pairwise_ttests(merged_dataframe, dv=parameter, between=[comparison_type, between_type], correction=True, parametric=False, padjust='bonf')
        df.rename(columns={'Contrast':'contrast', 'Parametric':'parametric', 'Tail':'tail', 'p-corr': 'p-val', 'p-adjust':'correction_method'}, inplace=True)
        df.drop(columns=['Paired', 'U-val', 'p-unc', 'hedges'], inplace=True)
        df['statistical_test'] = 'Mann-Whitney MC'
        df['significance'] = df['p-val'].apply(get_sig)
    else:
        df = pd.DataFrame([{y: np.nan for _, y in enumerate(col_order)}])
        df['contrast'] = comparison_type
    df['channel'] = channel
    df['attribute'] = attribute
    df['parameter'] = parameter
    df = df[col_order]
    return df