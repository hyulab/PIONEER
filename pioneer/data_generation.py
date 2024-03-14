import os
import glob
import gzip
import json
import pickle
import itertools
import pkg_resources
import numpy as np
import pandas as pd
from collections import defaultdict
from .config import *
from .nn_utils import *
from .utils import unzip_res_range
from .features import calculate_excluded_pdb_dict
import statistics
from .config import SASA_AF_PATH, AF_PRED_PATH

AF_sasa = pd.read_csv(SASA_AF_PATH, sep = '\t')

def generate_seq_seq_data(int_list, seq_feat_dir):
    """
    Generate data for the sequence-sequence model.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        seq_feat_dir (str): Path to the directory where sequence feature pickle files are stored.
        
    Returns:
        A list of dictionaries, each corresponding to an interaction. Each dictionary has the following fields:
        'complex' -> 'p1_p2', 'p1_feature' -> np.ndarray of p1 features, 'p2_feature' -> np.ndarray of p2 features, 
        'p1_sample' -> list of p1 residue indices (residue number - 1), 'p2_sample' -> list of p2 residue indices.
    
    """
    colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/sequence_feature_colmean.pkl'))[1:].values
    data = []
    for idx, (p1, p2) in enumerate(int_list):
        if not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2))):
            print('Missing p1 feature file for interaction %s_%s' % (p1, p2))
            continue
        if p1 != p2 and not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2))):
            print('Missing p2 feature file for interaction %s_%s' % (p1, p2))
            continue
        int_data = {}
        p1_feature = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2))) # load p1 features
        seq_colnames = p1_feature.columns
        p1_feature = np.nan_to_num(p1_feature.values, nan = colmeans)
        p1_feature = pd.DataFrame(p1_feature, columns = seq_colnames)
        #p1_feature = p1_feature.fillna(colmeans) # This is not working

        p1_sample = list(range(p1_feature.shape[0]))
        if p1 != p2: # if heterodimer
            p2_feature = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2))) # load p2 features
            seq_colnames = p2_feature.columns
            p2_feature = np.nan_to_num(p2_feature.values, nan = colmeans)
            p2_feature = pd.DataFrame(p2_feature, columns = seq_colnames)
            #p2_feature = p2_feature.fillna(colmeans) # This is not working
            p2_sample = list(range(p2_feature.shape[0]))
        else:
            p2_feature, p2_sample = p1_feature, p1_sample
        if len(p1_sample) == 0 and len(p2_sample) == 0:
            print('Feature matrices are empty for both proteins for interaction %s_%s' % (p1, p2))
            continue
        int_data['complex'] = p1 + '_' + p2
        int_data['p1_feature'], int_data['p2_feature'] = p1_feature.values, p2_feature.values
        int_data['p1_sample'], int_data['p2_sample'] = p1_sample, p2_sample
        data.append(int_data)
    return data


def generate_models_to_use(int_list, seq_dict, excluded_pdb_dict, cached_prots, non_cached_dir, min_coverage=0.1):
    """
    Generate structural models to use for each interaction.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        excluded_pdb_dict (dict): Dictionary mapping each interaction to a list of PDB structures to exclude.
        cached_prots: The modbase information of these proteins have been cached.
        non_cached_dir (str): Output directory for each specific feature.
        min_coverage (float): Minimum coverage required for including a structure.
        
    Returns:
        A dictionary mapping (p1, p2) -> {p1: list_p1, p2: list_p2}, where `list_p1` and `list_p2` are lists of
        structures to use for p1 and p2, respectively.
    
    """
    # Load the human ModBase index file. This file should not be empty. If there is a problem with reading empty human modbase
    # index file, you may modify this part. But please make sure that everything below runs correctly, since this dataframe is
    # going to be concatenated with other dataframes. Make sure that the columns are correct.
    
    modbase_index = pd.DataFrame()
    if os.path.exists(os.path.join(non_cached_dir, 'modbase', 'parsed_files', 'select_modbase_models.txt')):
        modbase_index = pd.read_csv(os.path.join(non_cached_dir, 'modbase', 'parsed_files', 'select_modbase_models.txt'), sep='\t')
        modbase_index['source'] = 'MB'
        modbase_index['file'] = modbase_index['modbase_modelID'] + '.pdb.gz'
        modbase_index['uniprot_res'] = modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        modbase_index['model_res'] = modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        modbase_index['coverage'] = modbase_index['template_length'] / modbase_index['target_length']
        modbase_index = modbase_index.drop(['template_length', 'target_length', 'target_begin', 'target_end', 'model_score', 'modbase_modelID', 'eVALUE'], axis=1)
        
    human_modbase_index = pd.DataFrame()
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt')):
        human_modbase_index = pd.read_csv(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt'), sep='\t')
        human_modbase_index['source'] = 'MB'
        human_modbase_index['file'] = human_modbase_index['modbase_modelID'] + '.pdb.gz'
        human_modbase_index['uniprot_res'] = human_modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        human_modbase_index['model_res'] = human_modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        human_modbase_index['coverage'] = human_modbase_index['template_length'] / human_modbase_index['target_length']
        human_modbase_index = human_modbase_index.drop(['template_length', 'target_length', 'target_begin', 'target_end', 'model_score', 'modbase_modelID', 'eVALUE'], axis=1)
        human_modbase_index = human_modbase_index[human_modbase_index['uniprot'].isin(cached_prots)]
    
    # Load the non-human ModBase index file. This file keeps updating as new ModBase structures are downloaded.
    nonhuman_modbase_index = pd.DataFrame()
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt')):
        nonhuman_modbase_index = pd.read_csv(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt'), sep='\t')
        nonhuman_modbase_index['source'] = 'MB'
        nonhuman_modbase_index['file'] = nonhuman_modbase_index['modbase_modelID'] + '.pdb.gz'
        nonhuman_modbase_index['uniprot_res'] = nonhuman_modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        nonhuman_modbase_index['model_res'] = nonhuman_modbase_index.apply(lambda x: '[%s-%s]' %(x['target_begin'], x['target_end']), axis=1)
        nonhuman_modbase_index['coverage'] = nonhuman_modbase_index['template_length'] / nonhuman_modbase_index['target_length']
        nonhuman_modbase_index = nonhuman_modbase_index.drop(['template_length', 'target_length', 'target_begin', 'target_end', 'model_score', 'modbase_modelID', 'eVALUE'], axis=1)
        nonhuman_modbase_index = nonhuman_modbase_index[nonhuman_modbase_index['uniprot'].isin(cached_prots)]
        
    # get all PDB file names
    allPDBs = []
    PDB_codes = os.listdir(PDB_DATA_DIR)
    for code in PDB_codes:
        files = glob.glob(PDB_DATA_DIR + code + '/*.ent.gz')
        allPDBs += files
    allPDBs = list(set(allPDBs))
    allPDB_files = []
    for PDB in allPDBs:
        allPDB_files.append(os.path.basename(PDB))
    allPDB_files = set(allPDB_files)
    
    # Load the PDB residue mapping file
    pdb_model_index = pd.read_csv(PDBRESMAP_PATH, sep='\t')
    pdb_model_index['source'] = 'PDB'
    pdb_model_index['file'] = pdb_model_index.apply(lambda x: 'pdb' + x['PDB'].lower() + '.ent.gz', axis=1)
    pdb_model_index = pdb_model_index.drop(['UniProtRes', 'UniProtResSecondaryStructure', 'AllResInPDBChainOnPDBBasis'], axis=1)
    pdb_model_index.rename(columns={'MappableResInPDBChainOnUniprotBasis': 'uniprot_res', 'MappableResInPDBChainOnPDBBasis': 'model_res', 'UniProt': 'uniprot'}, inplace=True)
    pdb_model_index = pdb_model_index[pdb_model_index['file'].isin(allPDB_files)]
    
    # Concatenate the ModBase and PDB dataframes
    full_model_index = pd.concat([modbase_index, human_modbase_index, nonhuman_modbase_index, pdb_model_index], ignore_index=True, sort=True)
    full_model_index = full_model_index[full_model_index['uniprot'].isin(seq_dict)]
    
    # Remove models with small coverage
    if not full_model_index.empty:
        full_model_index['coverage'] = full_model_index.apply(lambda x: float(len(unzip_res_range(x['uniprot_res']))) / len(seq_dict[x['uniprot']]), axis=1) 
        full_model_index = full_model_index[full_model_index.coverage>=min_coverage]
    else:
        return {}
    
    col_order = ['uniprot', 'source', 'coverage', 'uniprot_res', 'model_res', 'file']
    col_order += [c for c in full_model_index.columns if c not in col_order]
    full_model_index = full_model_index[col_order]
    full_model_index = full_model_index.sort_values(['uniprot', 'source', 'coverage'], ascending=[True, False, False]) # Sort by model priority
    
    # Remove overlapping models
    def test(rows, threshold=5):
        info = []
        information_sets = [set(unzip_res_range(row)) for row in rows]
        cache = information_sets[0]
        valuable = [list(rows.keys())[0]]
        for i, r in zip(rows.keys(), information_sets):
            if i==0: continue
            if len(r-cache) > threshold:
                valuable.append(i)
                cache.update(r)
        return valuable
    
    indices = list(itertools.chain(*itertools.chain(*pd.DataFrame(full_model_index.groupby(by=['uniprot'])['uniprot_res'].apply(test)).values)))
    
    non_redundant = full_model_index.loc[indices]
    
    single_model_dict = defaultdict(list)
    if 'template_pdb' in non_redundant.columns.values.tolist():
        for _, row in non_redundant.iterrows():
            single_model_dict[row['uniprot']].append((row['source'], row['uniprot_res'], row['model_res'], row['file'], 
                                                      row['PDB'], row['Chain'], row['template_pdb'], row['template_chain']))
    else:
        for _, row in non_redundant.iterrows():
            single_model_dict[row['uniprot']].append((row['source'], row['uniprot_res'], row['model_res'], 
                                                      row['file'],row['PDB'], row['Chain']))
        
    models_to_use = {}
    for p1, p2 in int_list:
        to_exclude = excluded_pdb_dict[(p1, p2)]
        if p1 not in single_model_dict or p2 not in single_model_dict: #DL: No structure model for p1 or p2 or both
            models_to_use[(p1, p2)] = {}
        if p1 == p2: # Homodimer
            p1_usable = []
            for model in single_model_dict[p1]:
                if model[0] == 'PDB':
                    if model[4] not in to_exclude:
                        p1_usable.append(model)
                elif model[0] == 'MB':
                    if model[6].upper() not in to_exclude:
                        p1_usable.append(model)
            if p1_usable:
                models_to_use[(p1, p2)] = {p1: p1_usable, p2: p1_usable}
            else:
                models_to_use[(p1, p2)] = {}
        else: # Heterodimer
            p1_usable = []
            for model in single_model_dict[p1]:
                if model[0] == 'PDB':
                    if model[4] not in to_exclude:
                        p1_usable.append(model)
                elif model[0] == 'MB':
                    if model[6].upper() not in to_exclude:
                        p1_usable.append(model)
            p2_usable = []
            for model in single_model_dict[p2]:
                if model[0] == 'PDB':
                    if model[4] not in to_exclude:
                        p2_usable.append(model)
                elif model[0] == 'MB':
                    if model[6].upper() not in to_exclude:
                        p2_usable.append(model)
            if p1_usable and p2_usable:
                models_to_use[(p1, p2)] = {p1: p1_usable, p2: p2_usable}
            else:
                if p1_usable:
                    models_to_use[(p1, p2)] = {p1: p1_usable}
                elif p2_usable:
                    models_to_use[(p1, p2)] = {p2: p2_usable}
                else:
                    models_to_use[(p1, p2)] = {}
    return models_to_use

def get_chain(file_content, chain, mapping_dict, out_file):
    """
    Obtain information of the specific chain, transform PDB residue numbers into UniProt residue numbers, and
    write to the output file.
    
    Args:
        file_content (list): Content of the PDB file recorded by `readlines`.
        chain (char): Chain of interest.
        mapping_dict (dict): Dictionary mapping PDB residue numbers (in str) to UniProt residue numbers (in str).
        out_file (str): Path to the output structure file.
        
    Returns:
        None.
    
    """
    with open(out_file, 'w') as f:
        for l in file_content:
            l = l.decode('utf-8').strip()
            if l[:4] == 'ATOM' and l[21] == chain:
                if l[22:27].strip() not in mapping_dict.keys():
                    continue
                res_num = ' ' * (5 - len(mapping_dict[l[22:27].strip()])) + mapping_dict[l[22:27].strip()] # Fill in spaces as needed
                mapped_l = l[:22] + res_num + l[27:]
                f.write(mapped_l + '\n')

                
def correct_pdb(file):
    """
    Eliminate duplicates from a PDB structure file of a chain and write outputs back to an output file. For example,
    the output for the input file ***.out will be written in the same directory and will have the file name of ***.txt.
    The original input file will be removed.
    
    Args:
        file (str): Path to the file to correct. It should end in .out.
        
    Returns:
        None.
    
    """
    with open(file, 'r') as in_f:
        with open(file[:-4] + '.txt', 'w') as out_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    current_residue = line[22:27]
                try:
                    if int(line[22:27]) >= int(current_residue):
                        current_residue = line[22:27]
                        out_f.write(line)
                    if int(line[22:27]) < int(current_residue):
                        break
                except:
                    out_f.write(line)
    os.remove(file)
    
    
def get_mb(file_content, out_file):
    """
    Take lines starting with 'ATOM' from a ModBase file and write to an output file.
    """
    with open(out_file, 'w') as f:
        for l in file_content:
            if l[:4] == 'ATOM':
                f.write(l.strip() + '\n')


def get_contents(AF_path):
    f = gzip.open(AF_path, 'r')
    contents = f.readlines()
    contents = [i.strip().decode("utf-8") for i in contents]
    contents = filter(lambda x: x[:4] == "ATOM", contents)
    f.close()
    return list(contents)


def get_AF(AF_path, out_file):
    PDB = get_contents(AF_path)
    with open(out_file, 'w') as f:
        for l in PDB:
            f.write(l.strip() + '\n')


def get_res_id(structure_info):
    res_id = []
    for l in structure_info:
        res_id.append(l[22:28].strip())
    res_id = list(set(res_id))
    res_id = [int(i) for i in res_id]
    res_id.sort()
    return res_id


def get_quality(contents, res_id):
    quality = []
    for res in res_id:
        residue = list(filter(lambda x: int(x[22:28].strip()) == res, contents))
        quality.append(float(residue[0][60:67]))
    return quality

def get_ave_quality(UniProt):
    # All AF predictions
    """
    AF_predictions = []
    AF_prediction_path = '/fs/cbsuhyfs1/storage/resources/alphafold/data/'
    species = os.listdir(AF_prediction_path)
    for s in species:
        predictions = glob.glob(AF_prediction_path + s + '/*.pdb.gz')
        AF_predictions += predictions
    """
    with open(AF_PRED_PATH, 'rb') as f:
        AF_predictions = pickle.load(f)

    preds = list(filter(lambda x: UniProt in x, AF_predictions))
    preds.sort(key=lambda x: int(x.split('-')[2][1:]))
    quality_dict = {}
    res_keys = quality_dict.keys()
    for pred in preds:
        file_idx = int(pred.split('-')[2][1:])
        contents = get_contents(pred)
        res_id = get_res_id(contents)
        pred_quality = get_quality(contents, res_id)
        res_id = [i + 200*(file_idx - 1) for i in res_id]
        score_dict = dict(zip(res_id, pred_quality))
        for k, v in score_dict.items():
            if k in res_keys:
                quality_dict[k].append(v)
            else:
                quality_dict[k] = [v]
    ave_dict = {k:sum(v)/len(v) for (k,v) in quality_dict.items()}
    ave_scores = list(ave_dict.values())
    return ave_scores


def normalization(SASA_list):
    mean = sum(SASA_list)/len(SASA_list)
    std = statistics.stdev(SASA_list)
    SASA_list = [(x - mean)/std for x in SASA_list]
    return SASA_list


def get_SASA_feats(UniProt):
    sub_df = AF_sasa.loc[AF_sasa['UniProt'] == UniProt]
    f_names = sub_df['Structure'].values.tolist()
    f_names.sort(key=lambda x: int(x.split('-')[2][1:]))
    SASA_dict = {}
    residue_number = SASA_dict.keys()
    for f in f_names:
        SASA = sub_df.loc[sub_df['Structure'] == f]['SASA'].tolist()
        SASA = SASA[0].split(';')
        SASA = [float(x) for x in SASA]
        f_index = f.split('-')[2][1:]
        for i, v in enumerate(SASA):
            res_num = (int(f_index) - 1)*200 + i + 1
            if res_num in residue_number:
                SASA_dict[res_num].append(v)
            else:
                SASA_dict[res_num] = [v]

    MAX_SASA = {k:max(v) for k, v in SASA_dict.items()}
    MEAN_SASA = {k:sum(v)/len(v) for k, v in SASA_dict.items()}
    return MAX_SASA, MEAN_SASA


def add_AF_features(feature, UniProt, ave_quality, AF_threshold):
    MAX_SASA, MEAN_SASA = get_SASA_feats(UniProt)
    MAX_SASA = list(MAX_SASA.values())
    MAX_SASA_norm = normalization(MAX_SASA)
    MEAN_SASA = list(MEAN_SASA.values())
    MEAN_SASA_norm = normalization(MEAN_SASA)
    feature['SASA_combined_avg_raw'] = MEAN_SASA
    feature['SASA_combined_avg_norm'] = MEAN_SASA_norm
    feature['SASA_combined_max_raw'] = MAX_SASA
    feature['SASA_combined_max_norm'] = MAX_SASA_norm
    Hq_residues = np.where(np.array(ave_quality) > AF_threshold)[0]
    res_id = Hq_residues.tolist()
    res_id.sort()
    return feature, res_id


def get_struct_file(struct_info, non_cached_dir, out_file_prefix):
    """
    Obtain protein structure from file and write to an output file.
    
    Args:
        struct_info (list): a list of the format [struct_type, uniprot_range, pdb_range, filename, pdb_id, pdb_chain, 
            mb_template, mb_template_chain]. 
        non_cached_dir (str): Output directory for each specific feature.
        out_file_prefix (str): Prefix to the output structure file (everything except the extension). The actual
            extension depends on the structure type.
    
    Returns:
        None.
    
    """
    stype = struct_info[0]
    if stype == 'PDB':
        uniprot_range = struct_info[1]
        pdb_range = struct_info[2]
        mapping_dict = dict(zip(unzip_res_range(pdb_range), unzip_res_range(uniprot_range)))
        if not os.path.exists(os.path.join(PDB_DATA_DIR, struct_info[4][1:3].lower(), struct_info[3])):
            print(os.path.join(PDB_DATA_DIR, struct_info[4][1:3].lower(), struct_info[3]))
            return
        with gzip.open(os.path.join(PDB_DATA_DIR, struct_info[4][1:3].lower(), struct_info[3]), 'rb') as f:
            file_content = f.readlines()
        if os.path.exists(out_file_prefix + '_%s_%s.txt' % (struct_info[4], struct_info[5])):
            return
        get_chain(file_content, struct_info[5], mapping_dict, out_file_prefix + '_%s_%s.out' % (struct_info[4], struct_info[5]))
        correct_pdb(out_file_prefix + '_%s_%s.out' % (struct_info[4], struct_info[5]))

    elif stype == 'MB':
        if os.path.exists(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', struct_info[3][:-3])):
            with open(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', struct_info[3][:-3]), 'r') as f:
                file_content = f.readlines()
        elif os.path.exists(os.path.join(MODBASE_CACHE, 'models', 'hash', struct_info[3][:-3])):
            with open(os.path.join(MODBASE_CACHE, 'models', 'hash', struct_info[3][:-3]), 'r') as f:
                file_content = f.readlines()
        else:
            return
        if os.path.exists(out_file_prefix + '_%s_MB.txt' % struct_info[3][:-7]):
            return
        get_mb(file_content, out_file_prefix + '_%s_MB.txt' % struct_info[3][:-7])
    else: # type is AF prediction
        if os.path.exists(out_file_prefix + '_' + os.path.basename(struct_info[1])[:-10] + '_AF.txt'):
            return
        get_AF(struct_info[1], out_file_prefix + '_' + os.path.basename(struct_info[1])[:-10] + '_AF.txt')


def obtain_all_structures(int_list, models_to_use, non_cached_dir, out_dir):
    """
    Obtain all structures to be used for prediction and compile lists of interactions to be predicted by each 
    classifier. Fetched models will be stored in `out_dir`.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        models_to_use (dict): A dictionary generated by `generate_models_to_use` above containing information
            about all models to use for prediction.
        non_cached_dir (str): Output directory for each specific feature.
        out_dir (str): Path to directory to store fetched structure files.
    
    Returns:
        A tuple of 4 lists containing interactions to be predicted by the seq_seq, seq_str, str_seq and str_str
        classifiers, respectively.
    
    """
    seq_seq, seq_str, str_seq, str_str = [], [], [], []
    for idx, (p1, p2) in enumerate(int_list):
        seq_seq.append((p1, p2)) # Always use the seq_seq classifier
        if (p1, p2) not in models_to_use: # No structure available for the interaction
            continue
        model_dict = models_to_use[(p1, p2)] # Structural information of the interaction
        if p1 == p2: # Homodimer
            str_str.append((p1, p2))
            p1_models = model_dict[p1]
            for i, p1_model in enumerate(p1_models):
                get_struct_file(p1_model, non_cached_dir, os.path.join(out_dir, '_'.join([p1, p2, str(i), '0'])))
        else: # Heterodimer
            if len(model_dict) == 1: # One of p1 and p2 has structural information
                p = list(model_dict.keys())[0]
                if p == p1: # Only p1 has structure information
                    str_seq.append((p1, p2))
                    p1_models = model_dict[p1]
                    for i, p1_model in enumerate(p1_models):
                        get_struct_file(p1_model, non_cached_dir, os.path.join(out_dir, '_'.join([p1, p2, str(i), '0'])))
                else: # Only p2 has structure information
                    seq_str.append((p1, p2))
                    p2_models = model_dict[p2]
                    for i, p2_model in enumerate(p2_models):
                        get_struct_file(p2_model, non_cached_dir, os.path.join(out_dir, '_'.join([p1, p2, str(i), '1'])))
            else: # Both proteins have structural information
                str_str.append((p1, p2))
                p1_models = model_dict[p1]
                for i, p1_model in enumerate(p1_models):
                    get_struct_file(p1_model, non_cached_dir, os.path.join(out_dir, '_'.join([p1, p2, str(i), '0'])))
                p2_models = model_dict[p2]
                for i, p2_model in enumerate(p2_models):
                    get_struct_file(p2_model, non_cached_dir, os.path.join(out_dir, '_'.join([p1, p2, str(i), '1'])))
    return seq_seq, seq_str, str_seq, str_str

    
def pdb_txt2array_gcn(struct_file):
    """
    Generate array representation of PDB chain for GCN input.
    
    Args:
        struct_file (str): Path to the structure file.
        
    Returns:
        A np.ndarray where each row is one atom in the PDB chain. The array has 10 columns corresponding to
        10 columns in the raw file.
    
    """
    num_lines = pd.read_csv(struct_file, header=None).shape[0]
    pdb_array = np.array(range(num_lines*10), dtype=object).reshape(num_lines, 10)
    with open(struct_file, 'r') as f:
        for i, line in enumerate(f):
            line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38], line[38:46], line[46:54]]
            line_list = [i.strip() for i in line_list]
            pdb_array[i] = line_list
    pdb_array = pdb_array[(np.where((pdb_array[:,3] == '') | (pdb_array[:,3] == 'A')))]
    return pdb_array


def generate_seq_str_data(int_list, seq_feat_dir, full_feat_dir, struct_dir, non_cached_dir, p2=False, min_res_num=11, AF_threshold=90):
    """
    Generate data for the sequence-structure model.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        seq_feat_dir (str): Path to the directory where sequence feature pickle files are stored.
        full_feat_dir (str): Path to the directory where full feature pickle files are stored.
        struct_dir (str): Path to the directory where extracted structures are stored.
        p2 (bool): Whether we are generating data for predictions on p2 (p1: str, p2: seq).
        min_res_num (int): Minimum number of residues required for the structure of p2.
        
    Returns:
        A list of dictionaries, each corresponding to an interaction.
    
    """
    seq_colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/sequence_feature_colmean.pkl'))[1:]
    full_colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/structure_feature_colmean.pkl'))[1:-1]
    if not p2: #DL: p2 is False. int_list is a list of interactions that only p2 has structure
        seq_poi, full_poi = '0', '1'
    else: #DL: p2 is True. int_list is a list of interactions that only p1 has structure (make predictions for residues in p2)
        seq_poi, full_poi = '1', '0'
    data = []
    for idx, (p1, p2) in enumerate(int_list):
        try:
            if seq_poi == '0': #DL: p2 = False (POI is p1)
                if not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2))):
                    continue
                if not os.path.exists(os.path.join(full_feat_dir, '%s_%s_1.pkl' % (p1, p2))): # Must be a heterodimer for the seq_str model
                    continue
            else: #DL: p2 = True (POI is p2)
                if not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2))):
                    continue
                if not os.path.exists(os.path.join(full_feat_dir, '%s_%s_0.pkl' % (p1, p2))):
                    continue
            
            # Obtain p1 features.
            p1_feature = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_%s.pkl' % (p1, p2, seq_poi))) #DL: If p2 is True, seq_poi is 1
            seq_colnames = p1_feature.columns
            p1_feature = np.nan_to_num(p1_feature.values, nan = seq_colmeans)
            p1_feature = pd.DataFrame(p1_feature, columns = seq_colnames)
            p1_sample = list(range(p1_feature.shape[0]))
            
            # Obtain the best p2 structure file.
            p2_struct_files = glob.glob(os.path.join(struct_dir, '%s_%s_*_%s_*.txt' % (p1, p2, full_poi))) #DL: If p2 is True, full_poi is 0
            
            if len(p2_struct_files) == 0:
                continue
            p2_struct_files.sort()
            p2_best_struct_file = p2_struct_files[0]
            struct_id, chain = os.path.basename(p2_best_struct_file)[:-4].split('_')[-2:]
            complex_code = '%s_%s_%s' % (p1, p2, seq_poi)
            
            # Extract residue numbers in the p2 structure.
            p2_pdb_array = pdb_txt2array_gcn(p2_best_struct_file)
            p2_res_id = [(int(x) - 1) for x in np.unique(p2_pdb_array[:, 6])]
            p2_res_id.sort()
            if len(p2_res_id) < min_res_num:
                continue
                
            # Generate edge indices. 
            if chain == 'MB':
                dm_fname = struct_id + '.npy'
            elif chain == 'AF':
                dm_fname = struct_id + '_v1.npy'
            else:
                dm_fname = struct_id + '_' + chain + '.npy'
            if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                p2_edge_index = get_edge_coo_data_pre_dm(p2_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
            else:
                p2_edge_index, new_dm = get_edge_coo_data(p2_pdb_array, 10)
                distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                if not os.path.exists(distance_matrix_dir):
                    os.system('mkdir -p '+distance_matrix_dir)
                np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
            
            # Generate feature arrays.
            p2_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_%s.pkl' % (p1, p2, full_poi)))
            p2_copy = p2
            if chain == 'AF':
                if full_poi == '0':
                    p2 = p1
                ave_quality = get_ave_quality(p2)
                if len(ave_quality) != p2_feature.shape[0]:
                    print("Size mis-match")
                    continue
                p2_feature, p2_res_id = add_AF_features(p2_feature, p2, ave_quality, AF_threshold)
                p2_res_id = list(range(0, p2_feature.shape[0]))
            p2 = p2_copy
            full_colnames = p2_feature.columns
            p2_feature = np.nan_to_num(p2_feature.values, nan = full_colmeans)
            p2_feature = pd.DataFrame(p2_feature, columns = full_colnames)
            p2_rnn_feature = p2_feature.values
            
            p2_gcn_feature = p2_feature.values[p2_res_id]

            int_data = {}
            int_data['complex'] = complex_code
            int_data['rnn_feature'] = p1_feature.values
            int_data['sample'] = p1_sample
            int_data['partner_edge_index'] = p2_edge_index
            int_data['partner_gcn_feature'] = p2_gcn_feature
            int_data['partner_rnn_feature'] = p2_rnn_feature
            int_data['partner_res_id'] = p2_res_id
            data.append(int_data)
        except:
            print('Error assemblying features for the seq_str model for interaction %s_%s' % (p1, p2))
            continue
    return data


def generate_str_seq_data(int_list, seq_feat_dir, full_feat_dir, struct_dir, non_cached_dir, p2=False, min_res_num=11, AF_threshold = 90):
    """
    Generate data for the structure-sequence model.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        seq_feat_dir (str): Path to the directory where sequence feature pickle files are stored.
        full_feat_dir (str): Path to the directory where full feature pickle files are stored.
        struct_dir (str): Path to the directory where extracted structures are stored.
        p2 (bool): Whether we are generating data for predictions on p2 (p1: seq, p2: str).
        min_res_num (int): Minimum number of residues required for the structure of p1.
        
    Returns:
        A list of dictionaries, each corresponding to an interaction. Each interaction could be represented by
        multiple dictionaries, corresponding to different structures of p1 that cover different residues.
    
    """
    seq_colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/sequence_feature_colmean.pkl'))[1:]
    full_colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/structure_feature_colmean.pkl'))[1:-1]
    if not p2: #DL: p2 is False
        seq_poi, full_poi = '1', '0'
    else: #DL: p2 is True. int_list is a list of interactions that only p2 has structure (make predictions for residues in p2)
        seq_poi, full_poi = '0', '1'
    data = []
    for p1, p2 in int_list:
        try:
            if seq_poi == '1':
                if not os.path.exists(os.path.join(full_feat_dir, '%s_%s_0.pkl' % (p1, p2))):
                    print('Missing p1 feature file for interaction %s_%s' % (p1, p2))
                    continue
                if not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_1.pkl' % (p1, p2))):
                    print('Missing p2 feature file for interaction %s_%s' % (p1, p2))
                    continue
            else: #DL: If p2 is True
                if not os.path.exists(os.path.join(full_feat_dir, '%s_%s_1.pkl' % (p1, p2))):
                    print('Missing p2 feature file for interaction %s_%s' % (p1, p2))
                    continue
                if not os.path.exists(os.path.join(seq_feat_dir, '%s_%s_0.pkl' % (p1, p2))):
                    print('Missing p1 feature file for interaction %s_%s' % (p1, p2))
                    continue

            # Obtain all p1 structure files.
            p1_struct_files = glob.glob(os.path.join(struct_dir, '%s_%s_*_%s_*.txt' % (p1, p2, full_poi))) #DL: If p2 is True, full_poi is 1
            if len(p1_struct_files) == 0:
                continue
            p1_struct_files.sort()
            covered = set()
            for p1_struct_f in p1_struct_files:
                _, _, priority, poi, struct_id, chain = os.path.basename(p1_struct_f)[:-4].split('_')
                complex_code = '%s_%s_%s_%s' % (p1, p2, priority, poi)
                if chain == 'MB':
                    dm_fname = struct_id + '.npy'
                elif chain == 'AF':
                    dm_fname = struct_id + '_v1.npy'
                else:
                    dm_fname = struct_id + '_' + chain + '.npy'

                # Extract residue numbers in the p1 structure.
                p1_pdb_array = pdb_txt2array_gcn(p1_struct_f)
                p1_res_id = [(int(x) - 1) for x in np.unique(p1_pdb_array[:, 6])]
                p1_res_id.sort()
                if len(p1_res_id) < min_res_num:
                    continue
                if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                    p1_edge_index = get_edge_coo_data_pre_dm(p1_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
                else:
                    p1_edge_index, new_dm = get_edge_coo_data(p1_pdb_array, 10)
                    distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                    if not os.path.exists(distance_matrix_dir):
                        os.system('mkdir -p '+distance_matrix_dir)
                    np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
                uncovered = set(p1_res_id) - covered # Residues not yet covered by previous structures
                uncovered_list = list(uncovered)
                uncovered_list.sort()
                if len(uncovered) == 0: # If the structure does not cover new residues, do not include
                    continue

                # Generate feature arrays.
                p1_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_%s.pkl' % (p1, p2, full_poi)))
                p1_copy = p1
                if chain == 'AF':
                    if full_poi == '1':
                        p1 = p2
                    ave_quality = get_ave_quality(p1)
                    if len(ave_quality) != p1_feature.shape[0]:
                        continue
                    p1_feature, p1_hq_res_id = add_AF_features(p1_feature, p1, ave_quality, AF_threshold)
                    p1_res_id = list(range(0, p1_feature.shape[0]))
                    uncovered = set(p1_hq_res_id) - covered # Residues not yet covered by previous structures
                    uncovered_list = list(uncovered)
                    uncovered_list.sort()
                    if len(uncovered) == 0: # If the structure does not cover new residues, do not include
                        continue
                p1 = p1_copy
                full_colnames = p1_feature.columns
                p1_feature = np.nan_to_num(p1_feature.values, nan = full_colmeans)
                p1_feature = pd.DataFrame(p1_feature, columns = full_colnames)
                p1_rnn_feature = p1_feature.values
                p1_gcn_feature = p1_feature.values[p1_res_id]
                p2_feature = pd.read_pickle(os.path.join(seq_feat_dir, '%s_%s_%s.pkl' % (p1, p2, seq_poi)))
                seq_colnames = p2_feature.columns
                p2_feature = np.nan_to_num(p2_feature.values, nan = seq_colmeans)
                p2_feature = pd.DataFrame(p2_feature, columns = seq_colnames)
                p2_rnn_feature = p2_feature.values

                int_data = {}
                covered = covered.union(set(p1_res_id)) # Updated covered residues
                int_data['complex'] = complex_code
                int_data['gcn_feature'] = p1_gcn_feature
                int_data['rnn_feature'] = p1_rnn_feature
                int_data['edge_index'] = p1_edge_index
                int_data['res_id'] = p1_res_id
                int_data['sample'] = uncovered_list
                int_data['partner_rnn_feature'] = p2_rnn_feature
                data.append(int_data)
        except:
            print('Error assemblying features for the seq_str model for interaction %s_%s' % (p1, p2))
            continue
    return data


def generate_str_str_data(int_list, full_feat_dir, struct_dir, non_cached_dir, min_res_num=11, AF_threshold = 90):
    """
    Generate data for the structure-structure model.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate features for.
        full_feat_dir (str): Path to the directory where full feature pickle files are stored.
        struct_dir (str): Path to the directory where extracted structures are stored.
        min_res_num (int): Minimum number of residues required for the structure of p1.
        
    Returns:
        A list of dictionaries, each corresponding to an interaction. Each interaction could be represented by
        multiple dictionaries, corresponding to different structures of p1 or p2 that cover different residues.
    
    """
    full_colmeans = pd.read_pickle(pkg_resources.resource_filename(__name__, 'data/mean_values/structure_feature_colmean.pkl'))[1:-1]
    data = []
    for p1, p2 in int_list:
        try:
            p1_struct_files = glob.glob(os.path.join(struct_dir, '%s_%s_*_0_*.txt' % (p1, p2)))
            if len(p1_struct_files) == 0:
                continue
            p1_struct_files.sort()
            if p1 != p2:
                p2_struct_files = glob.glob(os.path.join(struct_dir, '%s_%s_*_1_*.txt' % (p1, p2)))
                if len(p2_struct_files) == 0:
                    continue
                p2_struct_files.sort()
                p2_best_struct_file = p2_struct_files[0] # Only take the best p2 structure

            covered = set()
            for p1_struct_f in p1_struct_files:
                _, _, priority, poi, struct_id, chain = os.path.basename(p1_struct_f)[:-4].split('_')
                complex_code = '%s_%s_%s_%s' % (p1, p2, priority, poi)
                if chain == 'MB':
                    dm_fname = struct_id + '.npy'
                elif chain == 'AF':
                    dm_fname = struct_id + '_v1.npy'
                else:
                    dm_fname = struct_id + '_' + chain + '.npy'

                # Extract residue numbers in the p1 structure.
                p1_pdb_array = pdb_txt2array_gcn(p1_struct_f)
                p1_res_id = [(int(x) - 1) for x in np.unique(p1_pdb_array[:, 6])]
                p1_res_id.sort()
                if len(p1_res_id) < min_res_num:
                    continue
                if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                    p1_edge_index = get_edge_coo_data_pre_dm(p1_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
                else:
                    p1_edge_index, new_dm = get_edge_coo_data(p1_pdb_array, 10)
                    distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                    if not os.path.exists(distance_matrix_dir):
                        os.system('mkdir -p '+distance_matrix_dir)
                    np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
                uncovered = set(p1_res_id) - covered # Residues not yet covered by previous structures
                uncovered_list = list(uncovered)
                uncovered_list.sort()
                if len(uncovered) == 0: # If the structure does not cover new residues, do not include
                    continue

                # Generate p1 feature array.
                p1_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_0.pkl' % (p1, p2)))
                if chain == 'AF':
                    ave_quality = get_ave_quality(p1)
                    if len(ave_quality) != p1_feature.shape[0]:
                        continue
                    p1_feature, p1_hq_res_id = add_AF_features(p1_feature, p1, ave_quality, AF_threshold)
                    p1_res_id = list(range(0, p1_feature.shape[0]))
                    uncovered = set(p1_hq_res_id) - covered # Residues not yet covered by previous structures
                    uncovered_list = list(uncovered)
                    uncovered_list.sort()
                    if len(uncovered) == 0: # If the structure does not cover new residues, do not include
                        continue
                
                full_colnames = p1_feature.columns
                p1_feature = np.nan_to_num(p1_feature.values, nan = full_colmeans)
                p1_feature = pd.DataFrame(p1_feature, columns = full_colnames)
                p1_rnn_feature = p1_feature.values
                p1_gcn_feature = p1_feature.values[p1_res_id]

                # Generate p2 feature array.
                if p1 == p2: # For homodimers, same as p1
                    p2_edge_index, p2_gcn_feature, p2_rnn_feature, p2_res_id = p1_edge_index, p1_gcn_feature, p1_rnn_feature, p1_res_id
                else: # For heterodimers, process the best p2 structure
                    _, _, priority, poi, struct_id, chain = os.path.basename(p2_best_struct_file)[:-4].split('_')
                    if chain == 'MB':
                        dm_fname = struct_id + '.npy'
                    elif chain == 'AF':
                        dm_fname = struct_id + '_v1.npy'
                    else:
                        dm_fname = struct_id + '_' + chain + '.npy'

                    p2_pdb_array = pdb_txt2array_gcn(p2_best_struct_file)
                    p2_res_id = [(int(x) - 1) for x in np.unique(p2_pdb_array[:, 6])]
                    p2_res_id.sort()
                    if len(p2_res_id) < min_res_num:
                        continue
                    if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                        p2_edge_index = get_edge_coo_data_pre_dm(p2_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
                    else:
                        p2_edge_index, new_dm = get_edge_coo_data(p2_pdb_array, 10)
                        distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                        if not os.path.exists(distance_matrix_dir):
                            os.system('mkdir -p '+distance_matrix_dir)
                        np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
                    p2_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_1.pkl' % (p1, p2)))
                    if chain == 'AF':
                        ave_quality = get_ave_quality(p2)
                        if len(ave_quality) != p2_feature.shape[0]:
                            continue
                        p2_feature, p2_res_id = add_AF_features(p2_feature, p2, ave_quality, AF_threshold) # For partner protein, we use all AF residues if AF is the only option
                        p2_res_id = list(range(0, p2_feature.shape[0]))
                    full_colnames = p2_feature.columns
                    p2_feature = np.nan_to_num(p2_feature.values, nan = full_colmeans)
                    p2_feature = pd.DataFrame(p2_feature, columns = full_colnames)
                    p2_rnn_feature = p2_feature.values
                    p2_gcn_feature = p2_feature.values[p2_res_id]
                int_data = {}
                covered = covered.union(set(p1_res_id)) # Update covered residues
                int_data['complex'] = complex_code
                int_data['gcn_feature'] = p1_gcn_feature
                int_data['rnn_feature'] = p1_rnn_feature
                int_data['edge_index'] = p1_edge_index
                int_data['res_id'] = p1_res_id
                int_data['sample'] = uncovered_list
                int_data['partner_edge_index'] = p2_edge_index
                int_data['partner_gcn_feature'] = p2_gcn_feature
                int_data['partner_rnn_feature'] = p2_rnn_feature
                int_data['partner_res_id'] = p2_res_id
                data.append(int_data)

            # If heterodimer, we also need to generate data where p2 is the protein of interest.
            if p1 != p2:
                # Generate p1 feature arrays.
                p1_best_struct_file = p1_struct_files[0]
                _, _, priority, poi, struct_id, chain = os.path.basename(p1_best_struct_file)[:-4].split('_')
                if chain == 'MB':
                    dm_fname = struct_id + '.npy'
                elif chain == 'AF':
                    dm_fname = struct_id + '_v1.npy'
                else:
                    dm_fname = struct_id + '_' + chain + '.npy'

                p1_pdb_array = pdb_txt2array_gcn(p1_best_struct_file)
                p1_res_id = [(int(x) - 1) for x in np.unique(p1_pdb_array[:, 6])]
                p1_res_id.sort()
                if len(p1_res_id) < min_res_num:
                    continue
                if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                    p1_edge_index = get_edge_coo_data_pre_dm(p1_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
                else:
                    p1_edge_index, new_dm = get_edge_coo_data(p1_pdb_array, 10)
                    distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                    if not os.path.exists(distance_matrix_dir):
                        os.system('mkdir -p '+distance_matrix_dir)
                    np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
                p1_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_0.pkl' % (p1, p2)))
                if chain == 'AF':
                    ave_quality = get_ave_quality(p1)
                    if len(ave_quality) != p1_feature.shape[0]:
                        continue
                    p1_feature, p1_res_id = add_AF_features(p1_feature, p1, ave_quality, AF_threshold)
                    p1_res_id = list(range(0, p1_feature.shape[0]))
                full_colnames = p1_feature.columns
                p1_feature = np.nan_to_num(p1_feature.values, nan = full_colmeans)
                p1_feature = pd.DataFrame(p1_feature, columns = full_colnames)
                p1_rnn_feature = p1_feature.values
                p1_gcn_feature = p1_feature.values[p1_res_id]

                covered = set()
                for p2_struct_f in p2_struct_files:
                    _, _, priority, poi, struct_id, chain = os.path.basename(p2_struct_f)[:-4].split('_')
                    complex_code = '%s_%s_%s_%s' % (p1, p2, priority, poi)
                    if chain == 'MB':
                        dm_fname = struct_id + '.npy'
                    elif chain == 'AF':
                        dm_fname = struct_id + '_v1.npy'
                    else:
                        dm_fname = struct_id + '_' + chain + '.npy'

                    p2_pdb_array = pdb_txt2array_gcn(p2_struct_f)
                    p2_res_id = [(int(x) - 1) for x in np.unique(p2_pdb_array[:, 6])]
                    p2_res_id.sort()
                    if len(p2_res_id) < min_res_num:
                        continue
                    if os.path.exists(os.path.join(DISTANCE_MATRIX_CACHE, dm_fname)): # If distance matrix has been pre-calculated
                        p2_edge_index = get_edge_coo_data_pre_dm(p2_pdb_array, 10, os.path.join(DISTANCE_MATRIX_CACHE, dm_fname))
                    else:
                        p2_edge_index, new_dm = get_edge_coo_data(p2_pdb_array, 10)
                        distance_matrix_dir = os.path.join(non_cached_dir, 'distance_matrix')
                        if not os.path.exists(distance_matrix_dir):
                            os.system('mkdir -p '+distance_matrix_dir)
                        np.save(os.path.join(distance_matrix_dir, dm_fname), new_dm)
                    uncovered = set(p2_res_id) - covered
                    uncovered_list = list(uncovered)
                    uncovered_list.sort()
                    if len(uncovered) == 0:
                        continue

                    # Generate p2 feature array
                    p2_feature = pd.read_pickle(os.path.join(full_feat_dir, '%s_%s_1.pkl' % (p1, p2)))
                    if chain == 'AF':
                        ave_quality = get_ave_quality(p2)
                        if len(ave_quality) != p2_feature.shape[0]:
                            continue
                        p2_feature, p2_hq_res_id = add_AF_features(p2_feature, p2, ave_quality, AF_threshold)
                        p2_res_id = list(range(0, p2_feature.shape[0]))
                        uncovered = set(p2_hq_res_id) - covered # Residues not yet covered by previous structures
                        uncovered_list = list(uncovered)
                        uncovered_list.sort()
                        if len(uncovered) == 0: # If the structure does not cover new residues, do not include
                            continue

                    full_colnames = p2_feature.columns
                    p2_feature = np.nan_to_num(p2_feature.values, nan = full_colmeans)
                    p2_feature = pd.DataFrame(p2_feature, columns = full_colnames)
                    p2_rnn_feature = p2_feature.values
                    p2_gcn_feature = p2_feature.values[p2_res_id]
                    int_data = {}
                    covered = covered.union(set(p2_res_id)) # Update covered residues
                    int_data['complex'] = complex_code
                    int_data['gcn_feature'] = p2_gcn_feature
                    int_data['rnn_feature'] = p2_rnn_feature
                    int_data['edge_index'] = p2_edge_index
                    int_data['res_id'] = p2_res_id
                    int_data['sample'] = uncovered_list
                    int_data['partner_edge_index'] = p1_edge_index
                    int_data['partner_gcn_feature'] = p1_gcn_feature
                    int_data['partner_rnn_feature'] = p1_rnn_feature
                    int_data['partner_res_id'] = p1_res_id
                    data.append(int_data)
        except:
            print('Error assemblying features for the str_str model for interaction %s_%s' % (p1, p2))
            continue
    return data
