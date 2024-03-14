import os
import re
import glob
import gzip
import time
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from urllib.request import urlopen
from scipy.spatial.distance import squareform
from .config import PDB_DATA_DIR


def calc_expasy(seq, feature, expasy_dict):
    """
    Calculate ExPaSy feature for a protein sequence.
    
    Args:
        seq (str): The sequence of the protein.
        feature (str): The feature to calculate.
        expasy_dict (dict): Dictionary containing ExPaSy feature values.
        
    Returns:
        Array of specified ExPaSy feature values of the input sequence.
        
    """
    return np.array([expasy_dict[feature][x] if x in expasy_dict[feature] else 0 for x in seq])


def aggregate_sca(sca_file, seq_dict):
    """
    Calculate SCA feature values from an SCA result file.
    
    Args:
        sca_file (str): Path to the SCA result file.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
      
    Returns:
        A tuple of 6 arrays corresponding to 6 SCA features: p1_max, p2_max, p1_mean, p2_mean,
        p1_top10 and p2_top10.
        
    """
    id1, id2 = os.path.basename(sca_file).split('.')[0].split('_')[:2]
    len1, len2 = len(seq_dict[id1]), len(seq_dict[id2])
    with open(sca_file) as f:
        data = f.readlines()
    tot_len = int(data[-1].strip().split()[1])
    if tot_len != len1 + len2:
        return
    sca_mat = squareform(np.array([float(l.strip().split('\t')[2]) for l in data]))
    id1_id2_mat = sca_mat[:len1, len1:]
    meancoev = np.mean(id1_id2_mat)
    if id1 != id2:
        id1_max = np.max(id1_id2_mat, axis=1) / meancoev
        id2_max = np.max(id1_id2_mat, axis=0) / meancoev
        id1_mean = np.mean(id1_id2_mat, axis=1) / meancoev
        id2_mean = np.mean(id1_id2_mat, axis=0) / meancoev
        id1_top10 = np.mean(-np.partition(-sca_mat[:len1, len1:], 10)[:,:10], axis=1) / meancoev
        id2_top10 = np.mean((-np.partition(-sca_mat[:len1, len1:], 10, axis=0))[:10], axis=0) / meancoev
    else:
        id1_max = id2_max = np.max(id1_id2_mat, axis=0) / meancoev
        id1_mean = id2_mean = np.mean(id1_id2_mat, axis=0) / meancoev
        id1_top10 = id2_top10 = np.mean(-np.partition(-sca_mat[:len1, len1:], 10)[:,:10], axis=1) / meancoev
    return id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10


def split_dca(dca_file, out_dir):
    """
    Split the DCA result file into an MI file and a DI file.
    
    Args:
        dca_file (str): Path to the DCA result file.
        out_dir (str): Path to store the MI and DI files generated.
        
    Returns:
        None.
        
    """
    basename = os.path.basename(dca_file).split('.')[0]
    output_mi = open(os.path.join(out_dir, basename + '.mi'), 'w')
    output_di = open(os.path.join(out_dir, basename + '.di'), 'w')
    for line in open(dca_file, 'r'):
        i, j, mi, di = line.strip().split()
        output_mi.write('%s\t%s\t%s\n' % (i, j, mi))
        output_di.write('%s\t%s\t%s\n' % (i, j, di))
    output_mi.close()
    output_di.close()
    
    
def aggregate_dca(split_file, seq_dict):
    """
    Calculate DCA feature values from a split DCA file (MI or DI).
    
    Args:
        split_file (str): Path to the split DCA file.
        seq_dict (dict): Dictionary of protein sequences.
        
    Returns:
        A tuple of 6 arrays corresponding to 6 DCA features: p1_max, p2_max, p1_mean, p2_mean,
        p1_top10 and p2_top10.
    
    """
    id1, id2 = os.path.basename(split_file).split('.')[0].split('_')[:2]
    len1, len2 = len(seq_dict[id1]), len(seq_dict[id2])
    df = pd.read_csv(split_file, sep='\t', header=None, names=['resA', 'resB', 'Score']).set_index(['resA', 'resB']).unstack()
    if df.shape[0] != len1 + len2 - 1:
        print('Protein lengths unexpected. Skipping ', f)
        return
    df.columns = df.columns.levels[1].values
    df.index = df.index.values
    df[1] = np.nan
    df = df[sorted(df.columns)]
    df.loc[len(df)+1] = np.nan
    
    npdat = df.values[:len1, len1:]
    id1_max = np.nanmax(npdat, axis=1)
    id2_max = np.nanmax(npdat, axis=0)
    id1_mean = np.nanmean(npdat, axis=1)
    id2_mean = np.nanmean(npdat, axis=0)
    id1_top10 = np.mean(np.sort(npdat, axis=1)[:,-10:], axis=1)
    id2_top10 = np.mean(np.sort(npdat, axis=0)[-10:,:], axis=0)
    return id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10


def download_modbase(uniprot, out_dir, sleep_interval=3):
    """
    Download ModBase models of a UniProt.
    
    Args:
        uniprot (str): UniProt accession number.
        out_dir (str): Path of the directory to store the ModBase models.
        
    Returns:
        None.
    
    """
    if os.path.exists(os.path.join(out_dir, '%s.pdb' % uniprot)):
        return
    command = ['wget', '--quiet', 'http://salilab.org/modbase/retrieve/modbase/?databaseID=' + uniprot, '-O', os.path.join(out_dir, uniprot + '.pdb')]
    subprocess.call(command)
    time.sleep(sleep_interval)
    
    
def parse_modbase(modbase_download, hash_dir, header_dir, uniprot_length):
    """
    Create content and hashes for downloaded ModBase file.
    
    Args:
        modbase_download (str): Path to the ModBase file downloaded.
        hash_dir (str): Path to the ModBase hash directory.
        header_dir (str): Path to the ModBase header directory.
        uniprot_length (int): Length of the protein downloaded.
        
    Returns:
        A list of paths to the hash files (model files) generated.
        
    """
    uniprot = os.path.basename(modbase_download).split('.')[0]
    models = [m.replace('<content>', '').replace('</content>', '').strip() for m in re.findall(r'<content>.*?</content>', open(modbase_download).read(), flags=re.DOTALL)]
    hashes = []
    for model in models:
        target_length = [d.replace('TARGET LENGTH:', '').strip() for d in re.findall(r'TARGET LENGTH:.*', model)]
        if len(target_length) != 1:
            continue
        try:
            target_length = int(target_length[0])
        except:
            continue
        if target_length != uniprot_length: # target length must be the same as uniprot length
            continue
        hash_id = [d.replace('MODPIPE MODEL ID:', '').strip() for d in re.findall(r'MODPIPE MODEL ID:.*', model)]
        if len(hash_id) > 1:
            continue
        hash_id = hash_id[0]
        if os.path.exists(os.path.join(header_dir, hash_id + '.txt')):
            continue
        sequence_identity = [d.replace('SEQUENCE IDENTITY:', '').strip() for d in re.findall(r'SEQUENCE IDENTITY:.*', model)]
        if len(sequence_identity) != 1:
            continue
        model_score = [d.replace('MODEL SCORE:', '').strip() for d in re.findall(r'MODEL SCORE:.*', model)]
        if model_score == []:
            model_score = [d.replace('GA341 SCORE:', '').strip() for d in re.findall(r'GA341 SCORE:.*', model)]
        if len(model_score) != 1:
            continue
        MPQS = [d.replace('ModPipe Quality Score:', '').strip() for d in re.findall(r'ModPipe Quality Score:.*', model)]
        if MPQS == []:
            MPQS = [d.replace('MPQS:', '').strip() for d in re.findall(r'MPQS:.*', model)]
        if len(MPQS) != 1:
            continue
        zDOPE = [d.replace('zDOPE:', '').strip() for d in re.findall(r'zDOPE:.*', model)]
        if zDOPE == []:
            zDOPE = [d.replace('zDOPE SCORE:', '').strip() for d in re.findall(r'zDOPE SCORE:.*', model)]
        if len(zDOPE) != 1:
            continue
        evalue = [d.replace('EVALUE:', '').strip() for d in re.findall(r'EVALUE:.*', model)]
        if len(evalue) != 1:
            continue
        pdb = [d.replace('TEMPLATE PDB:', '').strip() for d in re.findall(r'TEMPLATE PDB:.*', model)]
        if len(pdb) != 1:
            continue
        chain = [d.replace('TEMPLATE CHAIN:', '').strip() for d in re.findall(r'TEMPLATE CHAIN:.*', model)]
        if len(chain) != 1:
            continue
        target_begin = [d.replace('TARGET BEGIN:', '').strip() for d in re.findall(r'TARGET BEGIN:.*', model)]
        if len(target_begin) != 1:
            continue
        target_end = [d.replace('TARGET END:', '').strip() for d in re.findall(r'TARGET END:.*', model)]
        if len(target_end) != 1:
            continue
        template_begin = [d.replace('TEMPLATE BEGIN:', '').strip() for d in re.findall(r'TEMPLATE BEGIN:.*', model)]
        if len(template_begin) != 1:
            continue
        template_end = [d.replace('TEMPLATE END:', '').strip() for d in re.findall(r'TEMPLATE END:.*', model)]
        if len(template_end) != 1:
            continue
        try:
            template_length = int(target_end[0]) - int(target_begin[0])
        except:
            continue
        # Write header file
        header = [uniprot, template_length, target_length, pdb[0], chain[0], target_begin[0], target_end[0], sequence_identity[0], model_score[0], MPQS[0], zDOPE[0], evalue[0], hash_id]
        header_handle = open(os.path.join(header_dir, hash_id + '.txt'), 'w')
        header_handle.write('\t'.join([str(i) for i in header]))
        header_handle.close()
        # Write hash file
        model_handle = open(os.path.join(hash_dir, hash_id + '.pdb'), 'w')
        model_handle.write(model)
        model_handle.close()
        hashes.append(os.path.join(hash_dir, hash_id + '.pdb'))
    return hashes
    
    
def fix_modbase(hash_file):
    """
    Fix index problems in the ModBase hash file. The file will be fixed in-place.
    
    Args:
        hash_file (str): Path to the ModBase hash file.
        
    Returns:
        None.
    
    """
    file_handle = open(hash_file, 'r')
    model_start = 1
    first_aa = True
    fixed_lines = []
    for line in file_handle:
        if 'REMARK 220 TARGET BEGIN:' in line:
            model_start = int(line.replace('REMARK 220 TARGET BEGIN:', '').strip())
        if line[:4] == 'ATOM':
            first_line_half = line[:26]
            second_line_half = line[26:]
            current_index = first_line_half[22:26]
            if first_aa:
                try:
                    original_first_index = int(current_index)
                except:
                    break
            if first_aa and int(current_index) == model_start:
                break
            first_line_half = first_line_half[:-1*len(current_index)] + ' '*len(current_index)
            new_index = str(int(current_index) + model_start - 1)
            first_line_half = first_line_half[:-1*len(new_index)] + new_index
            
            fixed_lines.append(first_line_half + second_line_half)
            first_aa = False
        else:
            fixed_lines.append(line)
    if first_aa:
        file_handle.close()
        return
    output_file = open(hash_file, 'w')
    output_file.writelines(fixed_lines)
    output_file.close()

    
def filter_modbase(header_dir, prot_set, selected_models_out_file):
    """
    Filter models and create a summary file.
    
    Args:
        header_dir (str): Directory that stores all header files to filter.
        prot_set (set): Set of protein identifiers to consider.
        selected_models_out_file (str): Path to the output summary file containing information of filtered models.
        
    Returns:
        None.
    
    """
    all_header_files = glob.glob(os.path.join(header_dir, '*.txt'))
    all_modbase_models = []
    selected_modbase_models = []
    # Full set
    for f in all_header_files:
        header = open(f).read().strip().split('\t')
        if len(header) == 13:
            all_modbase_models.append(header)
        else: #DL: if the length is 12 (chain information is missing)
            header.insert(4, '\t')
            all_modbase_models.append(header)
    
    all_modbase_models = [l for l in all_modbase_models if l != ['']]
    all_modbase_models.sort()
    # Selected set:
    # remove redundant models (same pdb, range of indices -- pick 1 model with hightest MPQS to represent them all)
    # remove models below thresholds for reliability (less strict than modbase prescribes in order to have decent model coverage)
    filter_dict = {}
    for l in all_modbase_models:
        uniprot, template_length, target_length, pdb, chain, begin, end, sequence_identity, model_score, MPQS, zDOPE, eVALUE, hash_ID = l
        if uniprot not in prot_set:
            continue
        try:
            MPQS, begin, end = float(MPQS), float(begin), float(end)
        except:
            continue
        if MPQS < 1.1:
            continue
        if (uniprot, pdb) not in filter_dict:
            filter_dict[(uniprot, pdb)] = []
        filter_dict[(uniprot, pdb)].append((begin, end, MPQS, hash_ID))
    selected_hash_IDs = set()
    selected_redundant_hash = set() # keep track of models completely contained within another model of the same uniprot/pdb with a lower MPQS
    for f in filter_dict:
        for m in filter_dict[f]:
            model_start1 = m[0]
            model_end1 = m[1]
            model_mpqs1 = m[2]
            model_hash1 = m[3]
            selected_hash_IDs.add(model_hash1)
            for m2 in filter_dict[f]:
                model_start2 = m2[0]
                model_end2 = m2[1]
                model_mpqs2 = m2[2]
                model_hash2 = m2[3]
                if model_hash1 == model_hash2:
                    continue
                if model_start1 >= model_start2 and model_end1 <= model_end2 and model_mpqs1 <= model_mpqs2:
                    selected_redundant_hash.add(model_hash1)
    selected_modbase_models = [header for header in all_modbase_models if header[-1] in selected_hash_IDs and header[-1] not in selected_redundant_hash]
    if selected_modbase_models:
        header = ['uniprot', 'template_length', 'target_length', 'template_pdb', 'template_chain', 'target_begin', 'target_end', 'sequence_identity', 'model_score', 'modpipe_quality_score', 'zDOPE', 'eVALUE', 'modbase_modelID']
        with open(selected_models_out_file, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for l in selected_modbase_models:
                f.write('\t'.join(l) + '\n')


def is_binary_file(filename):
    """
    Check if file is binary (adapted from http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python).
    Originally written by Michael J Meyer.
    
    Args:
        filename (str): Path to the file to check.
        
    Returns:
        A boolean value. True indicates that the file is binary, and false otherwise.
    
    """
    textchars = bytearray([7,8,9,10,12,13,27]) + bytearray(range(0x20, 0x100))
    return bool(open(filename, 'rb').read(1024).translate(None, textchars))


def natural_keys(text):
    """
    Natural sorting (adapted from http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside).
    Useful for sorting PDB residues that have letters in them. Originally written by Michael J Meyer.
    Example:
    >> my_list = ['1', '3', '2', '7', '2B']
    >> my_list.sort(key=natural_keys)
    ['1', '2', '2B', '3', '7']
    
    Args:
        text (str): A string in the list to be sorted.
        
    Returns:
        The natural key of the string.
        
    """
    def atoi(text): return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split('(\d+)', text)]


def open_pdb(structure):
    """
    Return an opened PDB file handle from STDIN, file, local PDB cache, or web. 
    Originally written by Michael Meyer.
    
    Args:
        structure (str): File name of input PDB file (.pdb or .ent), or gzipped pdb file (.gz), or 4-letter pdb ID. 
            If no argument given, reads structure on STDIN.
        
    Returns:
        A file handle for the structure.
    
    """
    # STDIN
    if "<open file '<stdin>', mode 'r' at" in str(structure):
        pdb_filehandle = structure
    # AS UNCOMPRESSED PDB FILE
    elif os.path.exists(structure) and is_binary_file(structure) == False:   #file exists and is a text-based file
        pdb_filehandle = open(structure, 'r')
    # AS GZIPPED PDB FILE
    elif os.path.exists(structure) and is_binary_file(structure) == True:    #file exists and is likely a gzipped file
        try:
            testopen = gzip.open(structure, 'r')
            testopen.readline()
            testopen.close()
            pdb_filehandle = gzip.open(structure, 'r')
        except IOError:
            raise IOError('Invalid structure file-type. Structure file must be a plain-text PDB file or a gzipped PDB file.')
    # AS PDB FILE FROM LOCAL COPY OF THE PDB -OR- FROM THE WEB
    elif len(structure) == 4:
        pdb_storage_path = os.path.join(PDB_DATA_DIR, '%s/pdb%s.ent.gz' %(structure[1:3].lower(), structure.lower()))
        # local file
        if os.path.exists(pdb_storage_path):
            pdb_filehandle = gzip.open(pdb_storage_path, 'r')
        else:
            try:
                pdb_filehandle = urlopen('http://www.rcsb.org/pdb/files/%s.pdb' % (structure.upper()))
            except HTTPError:
                raise ValueError('Invalid structure input: %s. Not found as local file, as PDB structure in %s, or on the web.' %(structure, PDB_DATA_DIR))
    else:
        raise ValueError('Invalid structure input: %s. Not found as local file, and wrong number of characters for direct PDB reference.' %(structure))
    return pdb_filehandle


def generate_excluded_pdb_dict(excluded_pdb_path, interactions):
    """
    Generate a dictionary containing information about PDB codes to exclude during feature calculation.
    
    Args:
        excluded_pdb_path (str): Path to the excluded PDB information file.
        interactions (list or set): A list or set of interactions to calculate features for.
        
    Returns:
        A dictionary mapping (p1, p2) -> set([pdb1, pdb2, ...]), which are PDB codes to exclude.
    
    """
    excluded_pdb_df = pd.read_csv(excluded_pdb_path, sep='\t')
    excluded_pdb_df = excluded_pdb_df[~pd.isnull(excluded_pdb_df['excludedPDBs'])]
    if not excluded_pdb_df.empty:
        excluded_pdb_df['Interaction'] = excluded_pdb_df.apply(lambda x: (x['UniProtA'], x['UniProtB']), axis=1)
        excluded_pdb_df = excluded_pdb_df[(excluded_pdb_df['hasCC'] == 'Y') | (excluded_pdb_df['Interaction'].isin(interactions))]
        excluded_pdb_df['excludedPDBs'] = excluded_pdb_df['excludedPDBs'].apply(lambda x: set(x.split(';')))
        excluded_pdbs = excluded_pdb_df.set_index('Interaction').to_dict()['excludedPDBs']
        excluded_pdbs = defaultdict(set, excluded_pdbs)
    else:
        excluded_pdbs = defaultdict(set)
    return excluded_pdbs


def generate_uniprot2chains(seq_dict, pdb_sasa_path, human_modbase_sasa_path, other_modbase_sasa_path, modbase_sasa_path):
    """
    Generate a dictionary that contains all SASA values calculated from structural models of each protein.
    
    Args:
        seq_dict (dict): Dictionary of protein sequences.
        pdb_sasa_path (str): SASA file calculated from PDB structures. ('SASA_perpdb_alltax.txt')
        human_modbase_sasa_path (str): SASA file calculated from human ModBase structures. ('SASA_modbase_human.txt')
        other_modbase_sasa_path (str): SASA file calculated from other ModBase structures. ('SASA_modbase_other.txt')
        modbase_sasa_path (str): SASA file calculated for user. ('SASA_modbase.txt')
    
    Returns:
        A dictionary mapping UniProt id -> 4-character PDB ID -> SASA array.
        
    """
    uniprot2chains = defaultdict(lambda: defaultdict(list))
    if os.path.exists(pdb_sasa_path):
        with open(pdb_sasa_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                pdb, _, uniprot, sasa = line.strip().split('\t')
                if uniprot in seq_dict:
                    sasa_array = np.array([float(r) if r != 'NaN' else np.nan for r in sasa.split(';')])
                    uniprot2chains[uniprot][pdb.upper()].append(sasa_array)
                    
    if os.path.exists(human_modbase_sasa_path):
        with open(human_modbase_sasa_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                uniprot, _, _, template_pdb, _, _, _, _, modpipe_quality_score, _, _, _, sasa = line.strip().split('\t')
                if uniprot in seq_dict:
                    if float(modpipe_quality_score) < 1.1:
                        continue
                    sasa_array = np.array([float(r) if r != 'NaN' else np.nan for r in sasa.split(';')])
                    uniprot2chains[uniprot][template_pdb.upper()].append(sasa_array)
                    
    if os.path.exists(other_modbase_sasa_path):
        with open(other_modbase_sasa_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                uniprot, _, _, template_pdb, _, _, _, _, modpipe_quality_score, _, _, _, sasa = line.strip().split('\t')
                if uniprot in seq_dict:
                    if float(modpipe_quality_score) < 1.1:
                        continue
                    sasa_array = np.array([float(r) if r != 'NaN' else np.nan for r in sasa.split(';')])
                    uniprot2chains[uniprot][template_pdb.upper()].append(sasa_array)
                    
    if os.path.exists(modbase_sasa_path):
        with open(modbase_sasa_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                uniprot, _, _, template_pdb, _, _, _, _, modpipe_quality_score, _, _, _, sasa = line.strip().split('\t')
                if uniprot in seq_dict:
                    if float(modpipe_quality_score) < 1.1:
                        continue
                    sasa_array = np.array([float(r) if r != 'NaN' else np.nan for r in sasa.split(';')])
                    uniprot2chains[uniprot][template_pdb.upper()].append(sasa_array)
    return uniprot2chains


def generate_docking_set(interactions, seq_dict, models_to_use, output_pdb_file, output_mb_file, output_mix_file):
    """
    Generate a file containing information regarding PDB structures to dock.
    
    Args:
        interactions (list or set): A list or set of interactions to calculate features for.
        seq_dict (dict): Dictionary mapping UniProt identifiers to sequences.
        models_to_use (dict): A dictionary containing information about all models to use for prediction.
        output_pdb_file (str): Path to the output PDB docking set file.
        output_mb_file (str): Path to the output MB docking set file.
        output_mix_file (str): Path to the output mix docking set file.
        
    Returns:
        A DataFrame containing information regarding PDB structures to dock (needed for generating
        the mixed docking set).
    
    """
    uniprot2pdb = {}
    uniprot2mb = {}
    uniprot2mix = {}
    for interaction in interactions:
        if interaction in models_to_use:
            p1, p2 = interaction
            if p1 not in models_to_use[interaction] or p2 not in models_to_use[interaction]:
                continue
                
            flag1, flag2, flag3 = False, False, False
                    
            for e1 in models_to_use[interaction][p1]:
                for e2 in models_to_use[interaction][p2]:
                    if e1[0] == 'PDB' and e2[0] == 'PDB':
                        if not flag1:
                            uniprot2pdb[interaction] = (e1[4]+'_'+e1[5], e2[4]+'_'+e2[5], len(unzip_res_range(e1[1])), len(unzip_res_range(e2[1])))
                            flag1 = True
                    elif e1[0] == 'MB' and e2[0] == 'MB':
                        if not flag2:
                            uniprot2mb[interaction] = (e1[3][:-7], e2[3][:-7], len(unzip_res_range(e1[1])), len(unzip_res_range(e2[1])))
                            flag2 = True
                    else:
                        if not flag3:
                            l1, l2 = False, False
                            if e1[0] == 'PDB':
                                subA = e1[4]+'_'+e1[5]
                                covA = len(unzip_res_range(e1[1]))
                                l1 = True
                            elif e1[0] == 'MB':
                                subA = e1[3][:-7]
                                covA = len(unzip_res_range(e1[1]))
                                l1 = True

                            if e2[0] == 'PDB':
                                subB = e2[4]+'_'+e2[5]
                                covB = len(unzip_res_range(e2[1]))
                                l2 = True
                            elif e2[0] == 'MB':
                                subB = e2[3][:-7]
                                covB = len(unzip_res_range(e2[1]))
                                l2 = True

                            if l1 and l2:
                                uniprot2mix[interaction] = (subA, subB, covA, covB)
                                flag3 = True
                        
    with open(output_pdb_file, 'w') as f:
        f.write('\t'.join(['ProtA', 'ProtB', 'SubA', 'SubB', 'CovA', 'CovB']) + '\n')
        for interaction in interactions:
            if interaction in uniprot2pdb:
                subA, subB, covA, covB = uniprot2pdb[interaction]
                f.write('%s\t%s\t%s\t%s\t%s\t%s\n' %(interaction[0], interaction[1], subA, subB, covA, covB))
                        
    with open(output_mb_file, 'w') as f:
        f.write('\t'.join(['ProtA', 'ProtB', 'SubA', 'SubB', 'CovA', 'CovB']) + '\n')
        for interaction in interactions:
            if interaction in uniprot2mb:
                subA, subB, covA, covB = uniprot2mb[interaction]
                f.write('%s\t%s\t%s\t%s\t%s\t%s\n' %(interaction[0], interaction[1], subA, subB, covA, covB))
                
    with open(output_mix_file, 'w') as f:
        f.write('\t'.join(['ProtA', 'ProtB', 'SubA', 'SubB', 'CovA', 'CovB']) + '\n')
        for interaction in interactions:
            if interaction in uniprot2mix:
                subA, subB, covA, covB = uniprot2mix[interaction]
                f.write('%s\t%s\t%s\t%s\t%s\t%s\n' %(interaction[0], interaction[1], subA, subB, covA, covB))
    
def extract_atoms(structure, out_file, chain_dict={}, chain_map={}, atoms=set(), record_types=set(['ATOM', 'HETATM']), write_end=False, fix_modbase=False):
    """
    Extract only specific residues from specific chains and specific atoms (i.e. backbone) within those residues.
    First written by Michael J Meyer, and edited by Dongjin Lee and Charles Liang.
    
    Args:
        structure (str): PDB file (plain text or gzipped), or PDB ID.
        out_file (str): Path to the output parsed PDB file.
        chain_dict (dict): Dictionary containing chains and residues to take from input file, 
            i.e. chain_dict = { 'A': set(['1','2','3','4','5','10','11','11B','12'...]), 'B': set([...]) }, 
            OR chain_dict can contain mapppings to new residue names: (i.e for mapping to UniProt residues),
            i.e. chain_dict = { 'A': {'1':'27', '2':'28', '3':'29', '4':'30', '5':'31'},  'B': {...} }.
        chain_map (dict): Dictionary mapping native chains to new chain IDs in `out_file`,
            i.e. chain_map = {'G': 'A', 'C': 'B'}  maps all chain G to chain A and all chain C to chain B,
            resulting chains will be alphabetically stored in output file.
        atoms (set): a set containing atom types to keep in final structure, i.e. atoms = set(['CA', 'C', 'N', O, ...]).
        record_types (set or dict): Types of records to keep (field 1 - i.e. ATOM, HETATM, ANISOU, etc.).
            Empty vars mean take all--i.e. chain_dict = {} means take all chains; chain_dict = {'A': set(), 'B': set()} 
            means take all residues in chains A and B; atoms = set() means take all atoms in whichever residues and chains 
            indicated in `chain_dict`; record_types = set() means take all record types.
        write_end (bool): True writes 'END' as the last line of the new file.
        fix_modbase (bool): True removes problematic fields that often show up in ModBase models. 
    
    Returns:
        None.
    
    """
    pdbinfile = open_pdb(structure)
    new_lines = defaultdict(list)
    integers = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for l in pdbinfile:
        if isinstance(l, bytes):
            l = l.decode('utf-8')
        recordName = l[:6].strip()
        if record_types != set() and recordName not in record_types:
            continue
        atomName = l[12:16].strip()
        chainID = l[21]
        resName = l[17:20].strip()
        resSeq = l[22:27].strip()
        new_line = l
        if chainID not in chain_dict and chain_dict != {}:
            continue
        if chain_dict != {} and len(chain_dict[chainID]) != 0:  
            if resSeq not in chain_dict[chainID]:
                continue
            if type(chain_dict[chainID] )== dict:
                if chain_dict[chainID][resSeq][-1] in integers:
                    new_resSeq = chain_dict[chainID][resSeq].rjust(4) + ' '
                else:
                    new_resSeq = chain_dict[chainID][resSeq].rjust(5)
                new_line = new_line[:22] + new_resSeq + new_line[27:]
        if atoms != set() and atomName not in atoms:
            continue
        if fix_modbase:
            new_line = new_line[:73] + ' ' * 4 + new_line[13] + ' ' * 3 + '\n'  
        if chainID in chain_map:
            new_line = new_line[:21] + chain_map[chainID] + new_line[22:]
            index_chain = chain_map[chainID]
        else:
            index_chain = chainID
        new_lines[index_chain].append(new_line)
    
    with open(out_file, 'w') as f:
        for chain in sorted(new_lines.keys()):
            for line in new_lines[chain]:
                f.write(line)
        if write_end: 
            f.write('END\n')


def extract_single_model(pdb_file, out_file):
    """
    Extract the first model from a PDB file.
    
    Args:
        pdb_file (str): Path to the original PDB file.
        out_file (str): Path to the file where output is to be written.
        
    Returns:
        None.
    
    """
    out_f = open(out_file, 'w')
    atom_seen = set()
    with open(pdb_file, 'r') as in_f:
        for line in in_f:
            asn = int(line[6:11].strip()) # Atom serial number
            chain = line[21]
            if (chain, asn) in atom_seen:
                continue
            out_f.write(line)
            atom_seen.add((chain, asn))
    out_f.close()
    

def unzip_res_range(res_range):
    """
    Converts ranges in the form: [2-210] or [3-45,47A,47B,51-67] into lists of strings including all numbers 
    in these ranges in order. Written by Michael J Meyer.
    
    Args:
        res_range (str): String representation of residue range.
        
    Returns:
        A list where consisted of residue numbers in string format.
    
    """
    res_ranges = res_range.strip()[1:-1].split(',')
    index_list = []
    for r in res_ranges:
        if re.match('.+-.+', r):
            a, b = r.split('-')
            index_list += [str(n) for n in range(int(a), int(b)+1)]
        else:
            index_list.append(r)

    if index_list == ['']:
        return []
    else:
        return index_list
    

def zip_res_range(seq):
    """
    Converts lists of residues as string in the form "1,2,2E,3,4,5,6,46,67,68,68A,68C,69,70" to zipped ranges
    in the form "[1-2,2E,3-6,46,67-68,68A,68C,69-70]". Written by Haoran Lee.
    
    Args:
        seq (string or list): A string (residues separated by comma) or list representation of residue numbers.
    
    Returns:
        A residue range in string representation.
    
    """

    if type(seq) == list:
        seq = ','.join(seq)

    seqout = []
    tempst = ''
    for h in range(seq.count(',') + 1):
        i = seq.split(',')[h]
        if i.isdigit() and h < seq.count(',') and str(int(i) + 1) == seq.split(',')[h+1]:
            if tempst == '':
                tempst = i
        else:
            if tempst == '':
                seqout.append(i)
            else:
                seqout.append(tempst + '-' + i)
                tempst = ''
    return '[' + ','.join(seqout) + ']'


def parse_dictionary_list(filename, header=None, delim='\t', max_keys=None, max_lines=None):
    """
    Uses header row in file as keys for dictionaries. Written by Michael J Meyer. Modified by Charles Liang.
    
    Args:
        filename (str): Path to the file to parse.
        header (list): Header of the table.
        delim (str): Delimiter between columns.
        max_keys (int): Maximum number of columns to parse.
        max_lines (int): Maximum number of lines to parse.
        
    Returns:
        A list of dictionaries where each entry corresponds to a line, mapping column names -> values.
    
    """
    with open(filename, 'r') as handle:
        if header is None:
            header_keys = handle.readline().strip().split(delim)
        else:
            header_keys = header
        if max_keys is not None:
            header_keys = header_keys[:max_keys]
        data = []
        line_num = 0
        for l in handle:
            if max_lines is not None and line_num == max_lines:
                break
            line = l.replace('\n', '').replace('\r', '').split(delim)
            cur_dict = {}
            for i, key in enumerate(header_keys):
                cur_dict[key] = line[i]
            data.append(cur_dict)
            line_num += 1
    return data


def rank(features, direction):
    '''
    Convert residue features to their ranks within given array (highest value is rank 1). Written by Michael J Meyer.

    Args:
        features (list or np.array): Array of features to rank.
        direction (str): Direction to rank, `hi2lo` or `lo2hi`.

    Returns:
        An array of ranked feature values.

    '''
    if direction == 'hi2lo':
        features = -1 * np.array(features, dtype=float)
    elif direction == 'lo2hi':
        features = np.array(features, dtype=float)
    finite_indices = np.where(np.isfinite(features))[0]
    features[finite_indices] = stats.rankdata(features[finite_indices], method='min')
    return features


def normalize(features):
    '''
    Normalize residue features, centered at 0 and scaled by standard deviation. Written by Michael J Meyer.

    Args:
        features (list or np.array): Array of features to normalize.

    Returns:
        An array of normalized feature values.

    '''
    return (features - np.nanmean(features)) / np.nanstd(features)
