# identify exclusive sets of interactions predicted by each model
import pandas as pd
import numpy as np
import glob
import os

def integration(seq_seq_prediction_dir, seq_str_prediction_dir, str_seq_prediction_dir, str_str_prediction_dir, PIONEER_prediction_dir):
    str_str = glob.glob(str_str_prediction_dir + '/*.npy')
    str_str_inters = []
    for inter in str_str:
        p1, p2, priority, poi = os.path.basename(inter).split('_')
        str_str_inters.append(p1 + '_' + p2)
    str_str_inters = set(str_str_inters) # all interactions predicted by str_str
    
    # str str model
    str_str_inters = list(str_str_inters)
    str_str_inters.sort()
    for p_idx, inter in enumerate(str_str_inters):
        inters = glob.glob(str_str_prediction_dir + '/' + inter + '*_0.npy')
        if inters: # if predictions (POI: 0) exist
            tot_res = 0
            for inte in inters:
                tot_res = tot_res + np.load(inte, allow_pickle=True).shape[0]

            pred_array = np.zeros((tot_res, 6), dtype = object)
            idx = 0
            for inte in inters:
                pred = np.load(inte, allow_pickle=True)
                pred_array[idx:idx + len(pred)] = pred
                idx = idx + len(pred)
        
            pred = pd.DataFrame(pred_array, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
            indices = pred['res_id'].values.tolist()
            p1, p2 = inter.split('_')
            if p1 == p2:
                seq_str_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_0.npy', allow_pickle=True)) # heterodimers are not predicted by seq_str
            else:
                if os.path.exists(seq_str_prediction_dir + '/' + inter + '_0.npy'): # it may not exist because num of residues in AF prediction is different from that of feature file
                    seq_str_pred = pd.DataFrame(np.load(seq_str_prediction_dir + '/' + inter + '_0.npy', allow_pickle=True))
                else:
                    seq_str_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_0.npy', allow_pickle=True))
            seq_str_pred.columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model']
            seq_str_pred = seq_str_pred.loc[~seq_str_pred['res_id'].isin(indices)]
            pred = pd.concat([pred, seq_str_pred])
            pred = pred.sort_values(by=['res_id'])
            pred = pred.reset_index(drop = True)
            pred.to_pickle(PIONEER_prediction_dir + '/' + inter + "_0.pkl")
     
            p1, p2 = inter.split('_')
            if p1 != p2:
                inters = glob.glob(str_str_prediction_dir + '/' + inter + '*_1.npy')
                if inters: # if predictions (POI: 1) exist
                    tot_res = 0
                    for inte in inters:
                        tot_res = tot_res + np.load(inte, allow_pickle=True).shape[0]
                
                    pred_array = np.zeros((tot_res, 6), dtype = object)
                    idx = 0
                    for inte in inters:
                        pred = np.load(inte, allow_pickle=True)
                        pred_array[idx:idx + len(pred)] = pred
                        idx = idx + len(pred)

                    pred = pd.DataFrame(pred_array, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
                    indices = pred['res_id'].values.tolist()
                    if os.path.exists(seq_str_prediction_dir + '/' + inter + '_1.npy'):
                        seq_str_pred = pd.DataFrame(np.load(seq_str_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True))
                    else:
                        seq_str_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True))
                    seq_str_pred.columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model']
                    seq_str_pred = seq_str_pred.loc[~seq_str_pred['res_id'].isin(indices)]
                    pred = pd.concat([pred, seq_str_pred])
                    pred = pred.sort_values(by=['res_id'])
                    pred = pred.reset_index(drop = True)
                    pred.to_pickle(PIONEER_prediction_dir + '/' + inter + "_1.pkl")

        else:
            p1, p2 = inter.split('_')
            if p1 != p2:
                inters = glob.glob(str_str_prediction_dir + '/' + inter + '*_1.npy')
                if inters: # if predictions (POI: 1) exist
                    tot_res = 0
                    for inte in inters:
                        tot_res = tot_res + np.load(inte, allow_pickle=True).shape[0]
                
                    pred_array = np.zeros((tot_res, 6), dtype = object)
                    idx = 0
                    for inte in inters:
                        pred = np.load(inte, allow_pickle=True)
                        pred_array[idx:idx + len(pred)] = pred
                        idx = idx + len(pred)
                
                    pred = pd.DataFrame(pred_array, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
                    indices = pred['res_id'].values.tolist()
                    if os.path.exists(seq_str_prediction_dir + '/' + inter + '_1.npy'):
                        seq_str_pred = pd.DataFrame(np.load(seq_str_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True))
                    else:
                        seq_str_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True))
                    seq_str_pred.columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model']
                    seq_str_pred = seq_str_pred.loc[~seq_str_pred['res_id'].isin(indices)]
                    pred = pd.concat([pred, seq_str_pred])
                    pred = pred.sort_values(by=['res_id'])
                    pred = pred.reset_index(drop = True)
                    pred.to_pickle(PIONEER_prediction_dir + '/' + inter + "_1.pkl")
    
    str_seq = glob.glob(str_seq_prediction_dir + '/*.npy')
    str_seq_inters = []
    for inter in str_seq:
        p1, p2, priority, poi = os.path.basename(inter).split('_')
        str_seq_inters.append(p1 + '_' + p2)
    str_seq_inters = set(str_seq_inters) # all interactions predicted by str_seq
    str_seq_inters = list(str_seq_inters.difference(set(str_str_inters)))
    str_seq_inters.sort()
    for p_idx, inter in enumerate(str_seq_inters): # for each interaction
        inters = glob.glob(str_seq_prediction_dir + '/' + inter + '*_0.npy') # collect prediction files for p1
        if inters: # If prediction exists
            tot_res = 0 # total number of residues covered by all structure information
            for inte in inters:
                tot_res = tot_res + np.load(inte, allow_pickle=True).shape[0]

            pred_array = np.zeros((tot_res, 6), dtype = object)
            idx = 0
            for inte in inters: # combine all the predictions
                pred = np.load(inte, allow_pickle=True)
                pred_array[idx:idx + len(pred)] = pred
                idx = idx + len(pred)

            pred = pd.DataFrame(pred_array, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
            indices = pred['res_id'].values.tolist()
            seq_seq_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_0.npy', allow_pickle=True)) # need seq_seq prediction for residues not covered by structure files
            seq_seq_pred.columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model']
            seq_seq_pred = seq_seq_pred.loc[~seq_seq_pred['res_id'].isin(indices)] # residues not covered by structure files
            pred = pd.concat([pred, seq_seq_pred])
            pred = pred.sort_values(by=['res_id'])
            pred = pred.reset_index(drop = True)
            pred.to_pickle(PIONEER_prediction_dir + '/' + inter + "_0.pkl")

        inters = glob.glob(str_seq_prediction_dir + '/' + inter + '*_1.npy') # collect prediction files for p2
        if inters: # If prediction exists
            tot_res = 0 # total number of residues covered by all structure information
            for inte in inters:
                tot_res = tot_res + np.load(inte, allow_pickle=True).shape[0]

            pred_array = np.zeros((tot_res, 6), dtype = object)
            idx = 0
            for inte in inters: # combine all the predictions
                pred = np.load(inte, allow_pickle=True)
                pred_array[idx:idx + len(pred)] = pred
                idx = idx + len(pred)

            pred = pd.DataFrame(pred_array, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
            indices = pred['res_id'].values.tolist()
            seq_seq_pred = pd.DataFrame(np.load(seq_seq_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True))
            seq_seq_pred.columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model']
            seq_seq_pred = seq_seq_pred.loc[~seq_seq_pred['res_id'].isin(indices)]
            pred = pd.concat([pred, seq_seq_pred])
            pred = pred.sort_values(by=['res_id'])
            pred = pred.reset_index(drop = True)
            pred.to_pickle(PIONEER_prediction_dir + '/' + inter + "_1.pkl")
    
    seq_str = glob.glob(seq_str_prediction_dir + '/*.npy')
    seq_str_inters = []
    for inter in seq_str:
        seq_str_inters.append(os.path.basename(inter)[:-6])
    seq_str_inters = set(seq_str_inters) # all interactions predicted by seq_str
    seq_str_inters = list(seq_str_inters.difference(set(str_str_inters)))
    seq_str_inters.sort()
    for idx, inter in enumerate(seq_str_inters):
        if os.path.exists(seq_str_prediction_dir + '/' + inter + '_0.npy'):
            pred = np.load(seq_str_prediction_dir + '/' + inter + '_0.npy', allow_pickle=True)
            pred = pd.DataFrame(pred, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
            pred.to_pickle(PIONEER_prediction_dir + '/'  + inter + "_0.pkl")
        if os.path.exists(seq_str_prediction_dir + '/' + inter + '_1.npy'):
            pred = np.load(seq_str_prediction_dir + '/' + inter + '_1.npy', allow_pickle=True)
            pred = pd.DataFrame(pred, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
            pred.to_pickle(PIONEER_prediction_dir + '/'  + inter + "_1.pkl")
    
    seq_seq = glob.glob(seq_seq_prediction_dir + '/*.npy')
    comb = glob.glob(PIONEER_prediction_dir + '/*.pkl')
    seq_seq_inter = []
    for interaction in seq_seq:
        seq_seq_inter.append(os.path.basename(interaction)[:-4])
    seq_seq_inter = set(seq_seq_inter)

    comb_inter = []
    for interaction in comb:
        comb_inter.append(os.path.basename(interaction)[:-4])
    comb_inter = set(comb_inter)

    diff = list(seq_seq_inter.difference(comb_inter))
    diff.sort()
    for prediction in diff:
        pred = np.load(seq_seq_prediction_dir + '/' + prediction + '.npy', allow_pickle=True)
        pred = pd.DataFrame(pred, columns = ['prot1', 'prot2', 'target_prot', 'res_id', 'prob', 'model'])
        pred.to_pickle(PIONEER_prediction_dir + '/' + prediction + ".pkl")
