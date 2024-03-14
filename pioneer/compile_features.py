import os
import numpy as np
import pandas as pd
from collections import defaultdict
from .features import *
from .msa import parse_fasta
from .data_generation import *
from .models_to_use_cleaning import *
from .AF_incorporation import *
from .config import UNIPROT_ALL_DB
from .utils import rank, normalize

def compile_pioneer_seq_feats(int_list, seq_dict, non_cached_dir, seq_feat_dir):
    """
    Calculate PIONEER sequence features and compile one dataframe in pickle format for each protein of an interaction.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        seq_feat_dir (str): Output directory for generated pickle files.
        
    Returns:
        None.
    
    """
    # Step 1: Calculate all sequence features
    expasy_feat_dict = calculate_expasy(seq_dict)
    print("Seq feat, Expasy finished")
    js_feat_dict = calculate_conservation(seq_dict, non_cached_dir)
    print("Seq feat, JS conservation finished")
    sca_feat_dict, dca_feat_dict = calculate_coevolution(int_list, seq_dict, non_cached_dir)
    print("Seq feat, SCA, DCA coevolution finished")
    raptorx_dict = calculate_raptorX(seq_dict, non_cached_dir)
    print("Seq feat, RaptorX finished")
    pp_dict, pp_norm_dict = calculate_pair_potential(int_list, seq_dict, non_cached_dir)
    print("Seq feat, PairPotential finished")

    # Step 2: Compile PIONEER sequence feature dataframes.
    for i in int_list:
        p1, p2 = i
        p1_features = pd.DataFrame()
        p2_features = pd.DataFrame()
        
        for feat in expasy_feat_dict: # ExPaSy features
            p1_features['expasy_%s_raw' % feat] = expasy_feat_dict[feat][p1]
            p2_features['expasy_%s_raw' % feat] = expasy_feat_dict[feat][p2]
            p1_features['expasy_%s_norm' % feat] = normalize(expasy_feat_dict[feat][p1])
            p2_features['expasy_%s_norm' % feat] = normalize(expasy_feat_dict[feat][p2])
            
        if p1 in js_feat_dict: # JS features
            p1_js_array = js_feat_dict[p1]
            p1_features['JS_raw'] = p1_js_array
            p1_features['JS_norm'] = normalize(p1_js_array)
        else:
            p1_features['JS_raw'], p1_features['JS_norm'] = [np.nan] * p1_features.shape[0], [np.nan] * p1_features.shape[0]

        if p2 in js_feat_dict:
            p2_js_array = js_feat_dict[p2]
            p2_features['JS_raw'] = p2_js_array
            p2_features['JS_norm'] = normalize(p2_js_array)
        else:
            p2_features['JS_raw'], p2_features['JS_norm'] = [np.nan] * p2_features.shape[0], [np.nan] * p2_features.shape[0]
            
        if i in sca_feat_dict['max']: # SCA and DCA features
            for ftype in sca_feat_dict:
                p1_sca_array, p2_sca_array = sca_feat_dict[ftype][i]
                p1_features['SCA_%s_raw' % ftype] = p1_sca_array
                p2_features['SCA_%s_raw' % ftype] = p2_sca_array
                p1_features['SCA_%s_norm' % ftype] = normalize(p1_sca_array)
                p2_features['SCA_%s_norm' % ftype] = normalize(p2_sca_array)
        else:
            for ftype in sca_feat_dict:
                p1_features['SCA_%s_raw' % ftype], p2_features['SCA_%s_raw' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
                p1_features['SCA_%s_norm' % ftype], p2_features['SCA_%s_norm' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            
        if i in dca_feat_dict['DMI_max']:
            for ftype in dca_feat_dict:
                p1_dca_array, p2_dca_array = dca_feat_dict[ftype][i]
                p1_features['%s_raw' % ftype] = p1_dca_array
                p2_features['%s_raw' % ftype] = p2_dca_array
                p1_features['%s_norm' % ftype] = normalize(p1_dca_array)
                p2_features['%s_norm' % ftype] = normalize(p2_dca_array)
        else:
            for ftype in dca_feat_dict:
                p1_features['%s_raw' % ftype], p2_features['%s_raw' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
                p1_features['%s_norm' % ftype], p2_features['%s_norm' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
                
        if p1 in raptorx_dict['SS_H_prob']: # RaptorX features
            for feat in raptorx_dict:
                p1_features[feat] = raptorx_dict[feat][p1]
                p1_features['%s_norm' % feat] = normalize(raptorx_dict[feat][p1])
        else:
            for feat in raptorx_dict:
                p1_features[feat], p1_features['%s_norm' % feat] = [np.nan] * p1_features.shape[0], [np.nan] * p1_features.shape[0]
        if p2 in raptorx_dict['SS_H_prob']:
            for feat in raptorx_dict:
                p2_features[feat] = raptorx_dict[feat][p2]
                p2_features['%s_norm' % feat] = normalize(raptorx_dict[feat][p2])
        else:
            for feat in raptorx_dict:
                p2_features[feat], p2_features['%s_norm' % feat] = [np.nan] * p2_features.shape[0], [np.nan] * p2_features.shape[0]
        
        if i in pp_dict: # Pair potential features
            p1_pp_array, p2_pp_array = pp_dict[i]
            p1_pp_norm, p2_pp_norm = pp_norm_dict[i]
            p1_features['pair_potential'], p2_features['pair_potential'] = p1_pp_array, p2_pp_array
            p1_features['pair_potential_norm'], p2_features['pair_potential_norm'] = p1_pp_norm, p2_pp_norm
        else:
            p1_features['pair_potential'], p2_features['pair_potential'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            p1_features['pair_potential_norm'], p2_features['pair_potential_norm'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
        
        feature_order = ['expasy_ACCE_raw', 'expasy_AREA_raw', 'expasy_BULK_raw', 'expasy_COMP_raw', 'expasy_HPHO_raw', 'expasy_POLA_raw', 'expasy_TRAN_raw', 'expasy_ACCE_norm', 'expasy_AREA_norm', 'expasy_BULK_norm', 'expasy_COMP_norm', 'expasy_HPHO_norm', 'expasy_POLA_norm', 'expasy_TRAN_norm', 'JS_raw', 'JS_norm', 'SCA_max_raw', 'SCA_max_norm', 'SCA_top10_raw', 'SCA_top10_norm', 'SCA_mean_raw', 'SCA_mean_norm', 'DMI_max_raw', 'DMI_max_norm', 'DMI_top10_raw', 'DMI_top10_norm', 'DMI_mean_raw', 'DMI_mean_norm', 'DDI_max_raw', 'DDI_max_norm', 'DDI_top10_raw', 'DDI_top10_norm', 'DDI_mean_raw', 'DDI_mean_norm', 'SS_H_prob', 'SS_E_prob', 'SS_C_prob', 'SS_H_prob_norm', 'SS_E_prob_norm', 'SS_C_prob_norm', 'ACC_B_prob', 'ACC_M_prob', 'ACC_E_prob', 'ACC_B_prob_norm', 'ACC_M_prob_norm', 'ACC_E_prob_norm', 'pair_potential', 'pair_potential_norm']
        p1_features[feature_order].to_pickle(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2)))
        if p1 != p2: 
            p2_features[feature_order].to_pickle(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2)))

def compile_pioneer_full_feats(int_list, seq_dict, non_cached_dir, seq_feat_dir, full_feat_dir, exclude_hom_struct=False):
    """
    Calculate all PIONEER features and compile one dataframe in pickle format for each protein of an interaction.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        seq_feat_dir (str): Output directory for generated pickle files.
        full_feat_dir (str): Output directory for generated pickle files.
        exclude_hom_struct (bool): Whether to exclude homologous interactions for structural features.
        
    Returns:
        None.
    
    """

    # Step 1: Calculate str features.
    
    # Only calculate structural features for UniProt sequences
    struct_int_list = [i for i in int_list if len(i[0]) != 64 and len(i[1]) != 64] # 64-character hash for unknown sequences
    struct_seq_dict = {k:v for k, v in seq_dict.items() if len(k) != 64}
    if exclude_hom_struct:
        excluded_pdb_dict = calculate_excluded_pdb_dict(struct_int_list, struct_seq_dict, non_cached_dir)
    else:
        excluded_pdb_dict = defaultdict(set)
    cached_prots, sasa_max_dict, sasa_mean_dict = calculate_sasa(struct_int_list, struct_seq_dict, excluded_pdb_dict, non_cached_dir)
    print("Str feat, SASA finished")
    
    models_to_use = generate_models_to_use(int_list, seq_dict, excluded_pdb_dict, cached_prots, non_cached_dir)
    models_to_use = models_to_use_cleaning(models_to_use)
    models_to_use = AF_incort(int_list, seq_dict, models_to_use)

    zdock_feat_dict = calculate_zdock(struct_int_list, struct_seq_dict, models_to_use, excluded_pdb_dict, non_cached_dir)
    print("Str feat, ZDOCK finished")
    
    # Step 2: Compile PIONEER full feature dataframes.
    for i in int_list:
        p1, p2 = i
        p1_features = pd.DataFrame()
        p2_features = pd.DataFrame()
        
        if i in sasa_mean_dict: # SASA features
            p1_sasa_array, p2_sasa_array = sasa_mean_dict[i]
            p1_features['SASA_combined_avg_raw'], p2_features['SASA_combined_avg_raw'] = p1_sasa_array, p2_sasa_array
            p1_features['SASA_combined_avg_norm'], p2_features['SASA_combined_avg_norm'] = normalize(p1_sasa_array), normalize(p2_sasa_array)
            p1_sasa_array, p2_sasa_array = sasa_max_dict[i]
            p1_features['SASA_combined_max_raw'], p2_features['SASA_combined_max_raw'] = p1_sasa_array, p2_sasa_array
            p1_features['SASA_combined_max_norm'], p2_features['SASA_combined_max_norm'] = normalize(p1_sasa_array), normalize(p2_sasa_array)
        else:
            p1_features['SASA_combined_avg_raw'], p2_features['SASA_combined_avg_raw'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            p1_features['SASA_combined_avg_norm'], p2_features['SASA_combined_avg_norm'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            p1_features['SASA_combined_max_raw'], p2_features['SASA_combined_max_raw'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            p1_features['SASA_combined_max_norm'], p2_features['SASA_combined_max_norm'] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
            
        if i in zdock_feat_dict['max']: # ZDOCK features
            for ftype in zdock_feat_dict:
                p1_zdock_array, p2_zdock_array = zdock_feat_dict[ftype][i]
                if ftype == 'mean':
                    ftype = 'avg'
                p1_features['zDOCK_dist3d_PRIORITY_0c_%s_raw' % ftype], p2_features['zDOCK_dist3d_PRIORITY_0c_%s_raw' % ftype] = p1_zdock_array, p2_zdock_array
                p1_features['zDOCK_dist3d_PRIORITY_0c_%s_norm' % ftype], p2_features['zDOCK_dist3d_PRIORITY_0c_%s_norm' % ftype] = normalize(p1_zdock_array), normalize(p2_zdock_array)
        else:
            for ftype in zdock_feat_dict:
                if ftype == 'mean':
                    ftype = 'avg'
                p1_features['zDOCK_dist3d_PRIORITY_0c_%s_raw' % ftype], p2_features['zDOCK_dist3d_PRIORITY_0c_%s_raw' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
                p1_features['zDOCK_dist3d_PRIORITY_0c_%s_norm' % ftype], p2_features['zDOCK_dist3d_PRIORITY_0c_%s_norm' % ftype] = [np.nan] * p1_features.shape[0], [np.nan] * p2_features.shape[0]
        
        feature_order = ['expasy_ACCE_raw', 'expasy_AREA_raw', 'expasy_BULK_raw', 'expasy_COMP_raw', 'expasy_HPHO_raw', 'expasy_POLA_raw', 'expasy_TRAN_raw', 'expasy_ACCE_norm', 'expasy_AREA_norm', 'expasy_BULK_norm', 'expasy_COMP_norm', 'expasy_HPHO_norm', 'expasy_POLA_norm', 'expasy_TRAN_norm', 'JS_raw', 'JS_norm', 'SCA_max_raw', 'SCA_max_norm', 'SCA_top10_raw', 'SCA_top10_norm', 'SCA_mean_raw', 'SCA_mean_norm', 'DMI_max_raw', 'DMI_max_norm', 'DMI_top10_raw', 'DMI_top10_norm', 'DMI_mean_raw', 'DMI_mean_norm', 'DDI_max_raw', 'DDI_max_norm', 'DDI_top10_raw', 'DDI_top10_norm', 'DDI_mean_raw', 'DDI_mean_norm', 'SASA_combined_avg_raw', 'SASA_combined_avg_norm', 'SASA_combined_max_raw', 'SASA_combined_max_norm', 'zDOCK_dist3d_PRIORITY_0c_min_raw', 'zDOCK_dist3d_PRIORITY_0c_max_raw', 'zDOCK_dist3d_PRIORITY_0c_avg_raw', 'zDOCK_dist3d_PRIORITY_0c_top1_raw', 'zDOCK_dist3d_PRIORITY_0c_min_norm', 'zDOCK_dist3d_PRIORITY_0c_max_norm', 'zDOCK_dist3d_PRIORITY_0c_avg_norm', 'zDOCK_dist3d_PRIORITY_0c_top1_norm', 'SS_H_prob', 'SS_E_prob', 'SS_C_prob', 'SS_H_prob_norm', 'SS_E_prob_norm', 'SS_C_prob_norm', 'ACC_B_prob', 'ACC_M_prob', 'ACC_E_prob', 'ACC_B_prob_norm', 'ACC_M_prob_norm', 'ACC_E_prob_norm', 'pair_potential', 'pair_potential_norm']
        
        p1_seq_features = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2)))
        p1_features = pd.concat([p1_seq_features, p1_features], axis=1)
        p1_features[feature_order].to_pickle(os.path.join(full_feat_dir, '%s_%s_0.pkl' % (p1, p2)))
        if p1 != p2: 
            p2_seq_features = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2)))
            p2_features = pd.concat([p2_seq_features, p2_features], axis=1)
            p2_features.to_pickle(os.path.join(full_feat_dir, '%s_%s_1.pkl' % (p1, p2)))
            
    return models_to_use
