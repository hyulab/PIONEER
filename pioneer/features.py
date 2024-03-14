import os
import glob
import json
import pickle
import shutil
import tempfile
import pkg_resources
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from .utils import *
from .zdock import *
from .config import *
from .pair_potential import *
from .msa import *
from .coevolution import sca, dca
from .js_divergence import calculate_js_div_from_msa
from .srescalc import calculate_SASA, gather_SASA, naccess

def calculate_expasy(seq_dict):
    """
    Calculate ExPaSy features.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
    
    Returns:
        A dictionary of dictionaries, feat -> protid -> feature array.
    
    """
    # Load ExPaSy data
    with open(pkg_resources.resource_filename(__name__, 'data/expasy.pkl'), 'rb') as f:
        expasy_dict = pickle.load(f)

    # Calculate ExPaSy features
    expasy_feat_dict = {}
    for feat in ['ACCE', 'AREA', 'BULK', 'COMP', 'HPHO', 'POLA', 'TRAN']:
        expasy_feat_dict[feat] = {}
        for prot in seq_dict:
            try:
                expasy_feat_dict[feat][prot] = calc_expasy(seq_dict[prot], feat, expasy_dict)
            except:
                print('Error calculating ExPaSy feature %s for %s' % (feat, prot))
                continue
    return expasy_feat_dict

def calculate_conservation(seq_dict, non_cached_dir):
    """
    Calculate JS divergence feature.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A dictionary mapping protein identifiers fo JS divergence feature arrays.
    """
    bg_distr = np.load(pkg_resources.resource_filename(__name__, 'data/bg_aa_distr.npy'))
    single_msa_file_dict = {} # id -> path
    js_feat_dict = {} # id -> js_array
    
    msa_files = []
    # Step 1: Generate single MSAs for all proteins queried.
    for prot, sequence in seq_dict.items():
        cached_js_file = os.path.join(JS_CACHE, '%s.npy' % prot)
        if os.path.exists(cached_js_file): # Retrieve results from cache if possible
            tmp_js_feat = np.load(cached_js_file)
            if len(tmp_js_feat) == len(seq_dict[prot]):
                js_feat_dict[prot] = tmp_js_feat
                continue
                
        cached_msa_file = os.path.join(MSA_CACHE, '%s.msa' % prot)
        if os.path.isfile(cached_msa_file): # Retrieve MSA if possible
            infile = open(cached_msa_file, 'r')
            lines = infile.readlines()
            infile.close()
            
            if len(lines[1].strip().replace('-', '')) == len(seq_dict[prot]):
                single_msa_file_dict[prot] = cached_msa_file
                continue
        
        # If no JS and MSA cached, generate MSA.
        if not os.path.exists(os.path.join(non_cached_dir, 'msa')):
            os.system('mkdir -p '+os.path.join(non_cached_dir, 'msa'))
            
        if os.path.exists(os.path.join(MSA_CACHE, '%s.rawmsa' % prot)):
            msa_file = os.path.join(MSA_CACHE, '%s.rawmsa' % prot)
        else:
            msa_file = generate_single_msa(prot, sequence, os.path.join(non_cached_dir, 'msa')) # Search for homologs using PSIBLAST
            
        msa_files.append([prot, sequence, msa_file])
        
    cdhit_input_file_dict = format_rawmsa(msa_files, os.path.join(non_cached_dir, 'msa'))
    for prot in cdhit_input_file_dict:
        cdhit_input_file = cdhit_input_file_dict[prot]
        if cdhit_input_file is None:
            continue
            
        cdhit_output_file = os.path.join(non_cached_dir, 'msa', prot+'.cdhit')
        run_cdhit(cdhit_input_file, cdhit_output_file)
        if not os.path.exists(cdhit_output_file):
            continue
            
        clustal_input_file = os.path.join(non_cached_dir, 'msa', prot+'.clustal_input')
        with open(cdhit_output_file, 'r') as infile:
            lines = infile.readlines()
        with open(clustal_input_file, 'w') as outfile:
            outfile.write('>'+prot+'\n'+seq_dict[prot]+'\n')
            for line in lines:
                outfile.write(line)
        
        clustal_output_file = os.path.join(non_cached_dir, 'msa', prot+'.clustal')
        run_clustal(clustal_input_file, clustal_output_file)
        if not os.path.exists(clustal_output_file):
            continue
            
        formatted_clustal_file = os.path.join(non_cached_dir, 'msa', prot+'.msa')
        format_clustal(clustal_output_file, formatted_clustal_file)
        if os.path.exists(formatted_clustal_file):
            single_msa_file_dict[prot] = formatted_clustal_file
    
    # Step 2: Calculate JS divergence from single MSAs generated.
    if not os.path.exists(os.path.join(non_cached_dir, 'js')):
        os.system('mkdir -p '+os.path.join(non_cached_dir, 'js'))
    for prot in single_msa_file_dict:
        try:
            js_array = calculate_js_div_from_msa(single_msa_file_dict[prot], bg_distr, 0.0000001, 3, 0.5) # PARAM
            js_feat_dict[prot] = js_array
        
            np.save(os.path.join(non_cached_dir, 'js', '%s.npy' % prot), js_array)
        except:
            print('Error calculating JS divergence for %s' % prot)
            continue
            
    return js_feat_dict

def calculate_coevolution(int_list, seq_dict, non_cached_dir):
    """
    Calculate coevolution features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A tuple of two dictionaries. The first dictionary maps (id1, id2) -> feat -> (sca1, sca2) and the second dictionary maps
        (id1, id2) -> feat -> (dca1, dca2).
    """
    sca_feat_dict = {'max': {}, 'mean': {}, 'top10': {}}
    dca_feat_dict = {'DDI_max': {}, 'DDI_mean': {}, 'DDI_top10': {}, 'DMI_max': {}, 'DMI_mean': {}, 'DMI_top10': {}}
    
    if not os.path.exists(os.path.join(non_cached_dir, 'coevolution')):
        os.system('mkdir -p '+os.path.join(non_cached_dir, 'coevolution'))
    
    int_list_ = []
    joined_msa_file_dict = {}
    for i in int_list:
        sca_fn = os.path.join(SCA_CACHE, '_'.join(i) + '.sca')
        dca_fn = os.path.join(DCA_CACHE, '_'.join(i) + '.dca')
        
        flag1, flag2 = False, False
        if os.path.exists(sca_fn):
            infile = open(sca_fn, 'r')
            lines = infile.readlines()
            infile.close()
            tot_len = int(lines[-1].strip().split()[1])
            if tot_len == len(seq_dict[i[0]])+len(seq_dict[i[1]]):
                flag1 = True
        if os.path.exists(dca_fn):
            infile = open(dca_fn, 'r')
            lines = infile.readlines()
            infile.close()
            tot_len = int(lines[-1].strip().split()[1])
            if tot_len == len(seq_dict[i[0]])+len(seq_dict[i[1]]):
                flag2 = True
            
        if flag1 and flag2:
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_sca(sca_fn, seq_dict)
            sca_feat_dict['max'][i] = (id1_max, id2_max)
            sca_feat_dict['mean'][i] = (id1_mean, id2_mean)
            sca_feat_dict['top10'][i] = (id1_top10, id2_top10)
            
            # Split DCA result into an MI file and a DI file
            split_dca(dca_fn, os.path.join(non_cached_dir, 'coevolution'))
            # Calculate DMI features from *.mi files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(non_cached_dir, 'coevolution', '_'.join(i) + '.mi'), seq_dict)
            dca_feat_dict['DMI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DMI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DMI_top10'][i] = (id1_top10, id2_top10)
            # Calculate DDI features from *.di files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(non_cached_dir, 'coevolution', '_'.join(i) + '.di'), seq_dict)
            dca_feat_dict['DDI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DDI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DDI_top10'][i] = (id1_top10, id2_top10)
        elif os.path.exists(os.path.join(MSA_CACHE, i[0]+'_'+i[1]+'.msa')):
            joined_msa_file_dict[i] = os.path.join(MSA_CACHE, i[0]+'_'+i[1]+'.msa')
        else:
            int_list_.append(i)
            
    prots = set()
    msa_fasta_file_dict = {}
    msa_files = []
    for i in int_list_:
        prots.add(i[0])
        prots.add(i[1])
    for prot in prots:
        if os.path.exists(os.path.join(non_cached_dir, 'msa', prot+'_rawmsa.fasta')):
            msa_fasta_file = os.path.join(non_cached_dir, 'msa', prot+'_rawmsa.fasta')
            msa_fasta_file_dict[prot] = msa_fasta_file
        else:
            if os.path.exists(os.path.join(MSA_CACHE, '%s.rawmsa' % prot)):
                msa_file = os.path.join(MSA_CACHE, '%s.rawmsa' % prot)
            else:
                if not os.path.exists(os.path.join(non_cached_dir, 'msa')):
                    os.system('mkdir -p '+os.path.join(non_cached_dir, 'msa'))
                msa_file = generate_single_msa(prot, seq_dict[prot], os.path.join(non_cached_dir, 'msa')) # Search for homologs using PSIBLAST
            msa_files.append([prot, seq_dict[prot], msa_file])
        
    msa_fasta_file_dict.update(format_rawmsa(msa_files, os.path.join(non_cached_dir, 'msa')))
    for i in int_list_:
        if msa_fasta_file_dict[i[0]] is None or msa_fasta_file_dict[i[1]] is None:
            continue
            
        cdhit_output_clstr_file1 = os.path.join(non_cached_dir, 'msa', i[0]+'.cdhit.clstr')
        cdhit_output_clstr_file2 = os.path.join(non_cached_dir, 'msa', i[1]+'.cdhit.clstr')
        if not os.path.exists(cdhit_output_clstr_file1):
            cdhit_output_file1 = os.path.join(non_cached_dir, 'msa', i[0]+'.cdhit')
            run_cdhit(msa_fasta_file_dict[i[0]], cdhit_output_file1)
            if not os.path.exists(cdhit_output_clstr_file1):
                continue
        if not os.path.exists(cdhit_output_clstr_file2):
            cdhit_output_file2 = os.path.join(non_cached_dir, 'msa', i[1]+'.cdhit')
            run_cdhit(msa_fasta_file_dict[i[1]], cdhit_output_file2)
            if not os.path.exists(cdhit_output_clstr_file2):
                continue
                
        clustal_input_file = cdhit_clstr_join(i[0], i[1], msa_fasta_file_dict[i[0]], msa_fasta_file_dict[i[1]], cdhit_output_clstr_file1, cdhit_output_clstr_file2, seq_dict, os.path.join(non_cached_dir, 'msa'))
        if clustal_input_file is None or not os.path.exists(clustal_input_file):
            continue
            
        clustal_output_file = os.path.join(non_cached_dir, 'msa', i[0]+'_'+i[1]+'.joined_clustal')
        run_clustal(clustal_input_file, clustal_output_file)
        if not os.path.exists(clustal_output_file):
            continue
            
        formatted_clustal_file = os.path.join(non_cached_dir, 'msa', i[0]+'_'+i[1]+'.joined_msa')
        format_clustal(clustal_output_file, formatted_clustal_file)
        if os.path.exists(formatted_clustal_file):
            joined_msa_file_dict[i] = formatted_clustal_file
            
    for i in joined_msa_file_dict:
        joined_msa = joined_msa_file_dict[i]
    
        # Step 2: Calculate SCA and DCA result files from joined MSA files generated.
        sca_fn = os.path.join(os.path.join(non_cached_dir, 'coevolution'), '_'.join(i) + '.sca')
        dca_fn = os.path.join(os.path.join(non_cached_dir, 'coevolution'), '_'.join(i) + '.dca')
        # Each protein must have more than 10 residues # PARAM
        if (len(seq_dict[i[0]]) > 10) and (len(seq_dict[i[1]]) > 10):
            # Don't need to recalculate if the file already exists
            try:
                sca(joined_msa, sca_fn)
            except:
                print('Error calculating SCA for %s' % '_'.join(i))
                continue

        if len(seq_dict[i[0]]) + len(seq_dict[i[1]]) <= 1000:
            # Maximum length of joined MSA supported for DCA is 1000 # PARAM
            # Don't need to recalculate if the file already exists
            try:
                dca(joined_msa, dca_fn)
            except:
                print('Error calculating DCA for %s' % '_'.join(i))
                continue

        # Step 3: Calculates final SCA and DCA results from their result files.
        if os.path.exists(sca_fn):
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_sca(sca_fn, seq_dict)
            sca_feat_dict['max'][i] = (id1_max, id2_max)
            sca_feat_dict['mean'][i] = (id1_mean, id2_mean)
            sca_feat_dict['top10'][i] = (id1_top10, id2_top10)
        
        if os.path.exists(dca_fn):
            # Split DCA result into an MI file and a DI file
            split_dca(dca_fn, os.path.join(non_cached_dir, 'coevolution'))
            # Calculate DMI features from *.mi files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(non_cached_dir, 'coevolution', '_'.join(i) + '.mi'), seq_dict)
            dca_feat_dict['DMI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DMI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DMI_top10'][i] = (id1_top10, id2_top10)
            # Calculate DDI features from *.di files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(non_cached_dir, 'coevolution', '_'.join(i) + '.di'), seq_dict)
            dca_feat_dict['DDI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DDI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DDI_top10'][i] = (id1_top10, id2_top10)
            
    return sca_feat_dict, dca_feat_dict

def calculate_excluded_pdb_dict(int_list, seq_dict, non_cached_dir):
    """
    Calculate dictionary containing PDBs to exclude for each interaction.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A dictionary mapping each interaction to a list of PDB structures to exclude.
    
    """
    # Step 1: Filter out query proteins with a UniProt ID.
    homolog_dir = os.path.join(non_cached_dir, 'homologs')
    if not os.path.exists(homolog_dir):
        os.system('mkdir -p '+homolog_dir)
        
    write_fasta(seq_dict, os.path.join(homolog_dir, 'query.fasta'))
    
    # Step 2: Build a sequence search database from the SIFTS info file.
    sifts_info_file = pkg_resources.resource_filename(__name__, 'data/sifts_uniprot_info.txt')
    sifts_dict = {}
    with open(sifts_info_file, 'r') as f:
        for line in f:
            if line.startswith('UniProt'):
                continue
            try:
                prot, _, _, _, _, seq = line.strip().split('\t') # There may be faulty lines
            except:
                continue
            sifts_dict[prot] = seq
    os.system('mkdir -p '+os.path.join(homolog_dir, 'db'))
    write_fasta(sifts_dict, os.path.join(homolog_dir, 'db', 'sifts.fasta'))
    os.system('%s -in %s -dbtype prot' % (MAKEBLASTDB, os.path.join(homolog_dir, 'db', 'sifts.fasta')))
    
    # Step 3: Run BLASTP for the query sequences against the SIFTS search database.
    if os.path.exists(os.path.join(homolog_dir, 'query.fasta')):
        os.system('%s -query %s -db %s -num_threads 8 -outfmt 6 > %s' % (BLASTP, os.path.join(homolog_dir, 'query.fasta'), os.path.join(homolog_dir, 'db', 'sifts.fasta'), os.path.join(homolog_dir, 'blastp_output.txt'))) # PARAM
    
    # Step 4: Parse BLASTP output and generate a dictionary specifying the homologs of all proteins.
    if os.path.exists(os.path.join(homolog_dir, 'blastp_output.txt')):
        homologs = defaultdict(set)
        with open(os.path.join(homolog_dir, 'blastp_output.txt'), 'r') as f:
            for line in f:
                query, target, _, _, _, _, _, _, _, _, e_value, _ = line.strip().split('\t')
                if float(e_value) < 1.0: # PARAM
                    homologs[query].add(target)
                    homologs[target].add(query)
        # Include proteins as their own homologs.
        for i in int_list:
            homologs[i[0]].add(i[0])
            homologs[i[1]].add(i[1])
            
        # Step 5: Build a file specifying excluded PDBs for each interaction.
        pdb2uniprots = defaultdict(set)  # Store all uniprots seen in each PDB
        pdbuniprot2count = defaultdict(int)  # Store number of times a uniprot is seen in each PDB
        uniprot2pdb = defaultdict(set)  # All pdbs associted with a uniprot and its homologs (reduce the set of uniprots to check for each interaction)
        with open(PDBRESMAP_PATH, 'r') as f:
            for line in f:
                if line.startswith('PDB'):
                    continue
                pdb, _, uniprot = line.strip().split('\t')[:3]
                pdb2uniprots[pdb].add(uniprot)
                pdbuniprot2count[(pdb, uniprot)] += 1
                homologs[uniprot].add(uniprot)
                for prot in homologs[uniprot]:
                    uniprot2pdb[prot].add(pdb)
        
        # Write the file specifying excluded PDBs.
        with open(os.path.join(homolog_dir, 'excluded_pdb.txt'), 'w') as f:
            f.write('\t'.join(['UniProtA', 'UniProtB', 'hasCC', 'excludedPDBs'])+'\n')
            for idx, i in enumerate(int_list):
                id1, id2 = i
                excluded_pdbs = set()
                has_CC = 'N'
                for pdb in uniprot2pdb[id1].union(uniprot2pdb[id2]):
                    if id1 == id2: # Homodimers
                        if pdbuniprot2count[(pdb, id1)] > 1:
                            excluded_pdbs.add(pdb)
                            has_CC = 'Y'
                        num_homologs_in_pdb = sum([pdbuniprot2count[(pdb, h)] for h in homologs[id1]])
                        if num_homologs_in_pdb > 1:
                            excluded_pdbs.add(pdb)
                    else: # Heterodimers
                        if id1 in pdb2uniprots[pdb] and id2 in pdb2uniprots[pdb]:
                            excluded_pdbs.add(pdb)
                            has_CC = 'Y'
                        if len(homologs[id1].intersection(pdb2uniprots[pdb])) > 0 and len(homologs[id2].intersection(pdb2uniprots[pdb])) > 0:
                            excluded_pdbs.add(pdb)
                f.write('%s\t%s\t%s\t%s\n' % (id1, id2, has_CC, ';'.join(sorted(excluded_pdbs))))
        excluded_pdb_dict = generate_excluded_pdb_dict(os.path.join(homolog_dir, 'excluded_pdb.txt'), int_list)
        return excluded_pdb_dict
    else:
        return defaultdict(set)

def calculate_sasa(int_list, seq_dict, excluded_pdb_dict, non_cached_dir):
    """
    Calculate SASA feature.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        excluded_pdb_dict (dict): Dictionary mapping each interaction to a list of PDB structures to exclude.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A tuple two dictionaries. The first maps (id1, id2) -> (sasa_max_1, sasa_max_2) and the second maps (id1, id2) -> (sasa_mean_1, sasa_mean_2).
    """
    
    prot_with_sasa = set() # Proteins that already have SASA information from ModBase
    uniprot2hash = {}
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt')):
        infile = open(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt'), 'r')
        lines = infile.readlines()[1:]
        infile.close()
        for line in lines:
            line_list = line.strip().split()
            if line_list[0] in uniprot2hash:
                uniprot2hash[line_list[0]].add(line_list[-1])
            else:
                uniprot2hash[line_list[0]] = set(line_list[-1])
                
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt')):
        infile = open(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt'), 'r')
        lines = infile.readlines()[1:]
        infile.close()
        for line in lines:
            line_list = line.strip().split()
            if line_list[0] in uniprot2hash:
                uniprot2hash[line_list[0]].add(line_list[-1])
            else:
                uniprot2hash[line_list[0]] = set(line_list[-1])
                
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt')):
        with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt'), 'r') as f:
            for i, line in enumerate(f):
                uniprot = line.strip().split()[0]
                if i > 0 and uniprot in seq_dict and len(seq_dict[uniprot]) == len(line.strip().split()[-1].split(';')) and uniprot in uniprot2hash:
                    for hashid in uniprot2hash[uniprot]:
                        if os.path.exists(os.path.join(MODBASE_CACHE, 'models', 'hash', '%s.pdb'%hashid)):
                            prot_with_sasa.add(uniprot)
                            break
                            
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt')):
        with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'), 'r') as f:
            for i, line in enumerate(f):
                uniprot = line.strip().split()[0]
                if i > 0 and uniprot in seq_dict and len(seq_dict[uniprot]) == len(line.strip().split()[-1].split(';')) and uniprot in uniprot2hash:
                    for hashid in uniprot2hash[uniprot]:
                        if os.path.exists(os.path.join(MODBASE_CACHE, 'models', 'hash', '%s.pdb'%hashid)):
                            prot_with_sasa.add(uniprot)
                            break
            
    modbase_unprot_dir = os.path.join(non_cached_dir, 'modbase', 'models', 'uniprot')
    modbase_hash_dir = os.path.join(non_cached_dir, 'modbase', 'models', 'hash')
    modbase_header_dir = os.path.join(non_cached_dir, 'modbase', 'models', 'header')
    modbase_parsed_dir = os.path.join(non_cached_dir, 'modbase', 'parsed_files')
    modbase_sasa_dir = os.path.join(non_cached_dir, 'modbase', 'SASA')
    
    # Step 1: download, parse, fix and filter ModBase models that do not exist in our folder, and create a summary file.
    for prot in seq_dict:
        if prot in prot_with_sasa:
            continue
            
        if not os.path.exists(os.path.join(modbase_parsed_dir, 'SASA_modbase.txt')):
            os.system('mkdir -p %s' % modbase_unprot_dir)
            os.system('mkdir -p %s' % modbase_hash_dir)
            os.system('mkdir -p %s' % modbase_header_dir)
            os.system('mkdir -p %s' % modbase_parsed_dir)
            os.system('mkdir -p %s' % modbase_sasa_dir)
            with open(os.path.join(modbase_parsed_dir, 'SASA_modbase.txt'), 'w') as f:
                f.write('\t'.join(['uniprot', 'template_length', 'target_length', 'template_pdb', 'target_begin', 'target_end', 'sequence_identity', 'model_score', 'modpipe_quality_score', 'zDOPE', 'eVALUE', 'modbase_modelID', 'SASA']) + '\n')
                
        # Download models from ModBase
        download_modbase(prot, modbase_unprot_dir)
        modbase_uniprot_pdb = os.path.join(modbase_unprot_dir, '%s.pdb' % prot) # prot_id.pdb in _MODBASE_CACHE/models/uniprot/
        if os.path.exists(modbase_uniprot_pdb):
            # Parse ModBase models
            model_hashes = parse_modbase(modbase_uniprot_pdb, modbase_hash_dir, modbase_header_dir, len(seq_dict[prot]))
            for hash_file in model_hashes:
                # Fix hash files
                fix_modbase(hash_file)
    
    # Filter modbase models and generate a summary file.
    filter_modbase(modbase_header_dir, set(seq_dict) - prot_with_sasa, os.path.join(modbase_parsed_dir, 'select_modbase_models.txt'))
    
    # Step 2: Calculate SASA for all selected models.
    model_path_list, length_list, header_info_list = [], [], []
    if os.path.exists(os.path.join(modbase_parsed_dir, 'select_modbase_models.txt')):
        selected_df = pd.read_csv(os.path.join(modbase_parsed_dir, 'select_modbase_models.txt'), sep='\t')
        for _, row in selected_df.iterrows():
            model_path_list.append(os.path.join(modbase_hash_dir, '%s.pdb' % row['modbase_modelID']))
            length_list.append(len(seq_dict[row['uniprot']]))
            header_info_list.append('\t'.join([str(x) for x in row.values]))
            
    pool = Pool(10) # PARAM
    pool.starmap(calculate_SASA, zip(model_path_list, length_list, header_info_list, [modbase_sasa_dir] * len(model_path_list)))
    pool.close()
    pool.join()
    
    # Step 3: Gather SASA values and calculate SASA features.
    gather_SASA(modbase_sasa_dir, os.path.join(modbase_parsed_dir, 'SASA_modbase.txt'))
    uniprot2chains = generate_uniprot2chains(seq_dict, SASA_PDB_PATH, os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt'), os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'), os.path.join(modbase_parsed_dir, 'SASA_modbase.txt'))
    
    sasa_max_dict, sasa_mean_dict = {}, {}
    for i in int_list:
        id1, id2 = i
        id1_sasas = []
        if id1 in uniprot2chains:
            for pdb in uniprot2chains[id1]:
                if pdb not in excluded_pdb_dict[i]:
                    id1_sasas += [sasas for sasas in uniprot2chains[id1][pdb] if len(sasas) == len(seq_dict[id1])]
        id2_sasas = []
        if id2 in uniprot2chains:
            for pdb in uniprot2chains[id2]:
                if pdb not in excluded_pdb_dict[i]:
                    id2_sasas += [sasas for sasas in uniprot2chains[id2][pdb] if len(sasas) == len(seq_dict[id2])]
                    
        if len(id1_sasas) == 0 and len(id2_sasas) == 0:
            continue
        if len(id1_sasas) == 0:
            id1_means, id1_max = [np.nan] * len(seq_dict[id1]), [np.nan] * len(seq_dict[id1])
        else:
            try:
                mdat = np.ma.masked_array(id1_sasas, np.isnan(id1_sasas))
                id1_means = np.mean(mdat, axis=0)
                id1_means = [id1_means.data[r] if id1_means.mask[r]==False else np.nan for r in range(len(id1_means.data))]
                id1_max = np.max(mdat, axis=0)
                id1_max = [id1_max.data[r] if id1_max.mask[r]==False else np.nan for r in range(len(id1_max.data))]
            except:
                continue
        if len(id2_sasas) == 0:
            id2_means, id2_max = [np.nan] * len(seq_dict[id2]), [np.nan] * len(seq_dict[id2])
        else:
            try:
                mdat = np.ma.masked_array(id2_sasas, np.isnan(id2_sasas))
                id2_means = np.mean(mdat, axis=0)
                id2_means = [id2_means.data[r] if id2_means.mask[r]==False else np.nan for r in range(len(id2_means.data))]
                id2_max = np.max(mdat, axis=0)
                id2_max = [id2_max.data[r] if id2_max.mask[r]==False else np.nan for r in range(len(id2_max.data))]
            except:
                continue
        if len(id1_means) != len(seq_dict[id1]) or len(id2_means) != len(seq_dict[id2]):
            continue
        sasa_max_dict[i] = (np.array(id1_max), np.array(id2_max))
        sasa_mean_dict[i] = (np.array(id1_means), np.array(id2_means))
        
    return prot_with_sasa, sasa_max_dict, sasa_mean_dict

def calculate_zdock(int_list, seq_dict, models_to_use, excluded_pdb_dict, non_cached_dir):
    """
    Calculate ZDOCK features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        models_to_use (dict): A dictionary containing information about all models to use for prediction.
        excluded_pdb_dict (dict): Dictionary mapping each interaction to a list of PDB structures to exclude.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A dictionary of the ZDOCK features mapping (id1, id2) -> feat -> (zdock1, zdock2).
    
    """
    # Step 1: Generate PDB, ModBase and mixed docking sets.
    os.system('mkdir -p %s' % os.path.join(non_cached_dir, 'zdock', 'docking_set'))
    pdb_docking_set_file = os.path.join(non_cached_dir, 'zdock', 'docking_set', 'pdb_docking_set.txt')
    #DL: generate_pdb_docking_set is defined in utils.py
    mb_docking_set_file = os.path.join(non_cached_dir, 'zdock', 'docking_set', 'mb_docking_set.txt')
    mixed_docking_set_file = os.path.join(non_cached_dir, 'zdock', 'docking_set', 'mixed_docking_set.txt')
    generate_docking_set(int_list, seq_dict, models_to_use, pdb_docking_set_file, mb_docking_set_file, mixed_docking_set_file)
    
    # Step 2: Perform Docking with ZDOCK.
    # Step 2.1: Perform PDB docking.
    pdb_docked_model_dir = os.path.join(non_cached_dir, 'zdock', 'docked_models', 'pdb_docked_models')
    if not os.path.exists(pdb_docked_model_dir):
        os.system('mkdir -p %s' % pdb_docked_model_dir)
            
    docking_df = pd.read_csv(pdb_docking_set_file, sep='\t')
    receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list = [], [], [], []
    pdb_receptor_ligand = set()
    flag = False
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']

        if os.path.exists(os.path.join(ZDOCK_CACHE, 'pdb_docked_models', 'PDB%s.ENT_%s--' % tuple(receptor.split('_')) + 'PDB%s.ENT_%s--ZDOCK.out' % tuple(ligand.split('_')))):
            pdb_receptor_ligand.add(os.path.join(ZDOCK_CACHE, 'pdb_docked_models', 'PDB%s.ENT_%s--' % tuple(receptor.split('_')) + 'PDB%s.ENT_%s--ZDOCK*.pdb' % tuple(ligand.split('_'))))
            continue
            
        receptor_pdb, receptor_chain = receptor.split('_')
        ligand_pdb, ligand_chain = ligand.split('_')
        receptor_pdb = os.path.join(PDB_DATA_DIR, receptor_pdb[1:-1].lower(), 'pdb%s.ent.gz' % receptor_pdb.lower())
        ligand_pdb = os.path.join(PDB_DATA_DIR, ligand_pdb[1:-1].lower(), 'pdb%s.ent.gz' % ligand_pdb.lower())
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
        receptor_chain_list.append(receptor_chain)
        ligand_chain_list.append(ligand_chain)
        
        pdb_receptor_ligand.add(os.path.join(pdb_docked_model_dir, 'PDB%s.ENT_%s--' % tuple(receptor.split('_')) + 'PDB%s.ENT_%s--ZDOCK*.pdb' % tuple(ligand.split('_'))))
        flag = True
        
    if flag:
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        my_pool = Pool(10) # PARAM
        my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list, [pdb_docked_model_dir] * len(receptor_pdb_list)))
        my_pool.close()
        my_pool.join()
    
    # Step 2.2: Perform ModBase docking.
    modbase_docked_model_dir = os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mb_docked_models')
    if not os.path.exists(modbase_docked_model_dir):
        os.system('mkdir -p %s' % modbase_docked_model_dir)
            
    docking_df = pd.read_csv(mb_docking_set_file, sep='\t')
    receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list = [], [], [], []
    modbase_receptor_ligand = set()
    
    flag = False
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']
            
        if os.path.isfile(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', receptor + '.pdb')):
            receptor_pdb = os.path.join(non_cached_dir, 'modbase', 'models', 'hash', receptor + '.pdb')
        else:
            receptor_pdb = os.path.join(MODBASE_CACHE, 'models', 'hash', receptor + '.pdb')
        receptor_all_chains = set()
        infile_receptor = open(receptor_pdb, 'r')
        for line in infile_receptor:
            if line.split()[0] == 'ATOM':
                receptor_all_chains.add(line[21])
        infile_receptor.close()
        if len(receptor_all_chains) != 1:
            continue
        receptor_all_chains = list(receptor_all_chains)
        if receptor_all_chains[0] == ' ':
            receptor_chain = '_'
        else:
            receptor_chain = receptor_all_chains[0]
            
        if os.path.isfile(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', ligand + '.pdb')):
            ligand_pdb = os.path.join(non_cached_dir, 'modbase', 'models', 'hash', ligand + '.pdb')
        else:
            ligand_pdb =  os.path.join(MODBASE_CACHE, 'models', 'hash', ligand + '.pdb')
        ligand_all_chains = set()
        infile_ligand = open(ligand_pdb, 'r')
        for line in infile_ligand:
            if line.split()[0] == 'ATOM':
                ligand_all_chains.add(line[21])
        infile_ligand.close()
        if len(ligand_all_chains) != 1:
            continue
        ligand_all_chains = list(ligand_all_chains)
        if ligand_all_chains[0] == ' ':
            ligand_chain = '_'
        else:
            ligand_chain = ligand_all_chains[0]
            
        if receptor_chain != '_':
            check_receptor = receptor.upper()+'_'+receptor_chain
        else:
            check_receptor = receptor.upper()
        if ligand_chain != '_':
            check_ligand = ligand.upper()+'_'+ligand_chain
        else:
            check_ligand = ligand.upper()
        
        if os.path.exists(os.path.join(ZDOCK_CACHE, 'modbase_docked_models', '%s--%s--ZDOCK.out' % (check_receptor.upper(), check_ligand.upper()))):
            modbase_receptor_ligand.add(os.path.join(ZDOCK_CACHE, 'modbase_docked_models', '%s--%s--ZDOCK*.pdb' % (check_receptor.upper(), check_ligand.upper())))
            continue
        
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
        receptor_chain_list.append(receptor_chain)
        ligand_chain_list.append(ligand_chain)
        
        modbase_receptor_ligand.add(os.path.join(modbase_docked_model_dir, '%s--%s--ZDOCK*.pdb' % (check_receptor.upper(), check_ligand.upper())))
        flag = True
        
    if flag:
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        my_pool = Pool(10) # PARAM
        my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list, [modbase_docked_model_dir] * len(receptor_pdb_list)))
        my_pool.close()
        my_pool.join()
        
    # Step 2.3: Perform Mixed docking.
    mixed_docked_model_dir = os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mixed_docked_models')
    if not os.path.exists(mixed_docked_model_dir):
        os.system('mkdir -p %s' % mixed_docked_model_dir)
            
    docking_df = pd.read_csv(mixed_docking_set_file, sep='\t')
    receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list = [], [], [], []
    mixed_receptor_ligand = set()
    flag = False
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']
            
        if len(receptor) == 32:  #modbase model IDs are 32 characters long
            if os.path.isfile(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', receptor + '.pdb')):
                receptor_pdb = os.path.join(non_cached_dir, 'modbase', 'models', 'hash', receptor + '.pdb')
            else:
                receptor_pdb = os.path.join(MODBASE_CACHE, 'models', 'hash', receptor + '.pdb')

            receptor_all_chains = set()
            infile_receptor = open(receptor_pdb, 'r')
            for line in infile_receptor:
                if line.split()[0] == 'ATOM':
                    receptor_all_chains.add(line[21])
            infile_receptor.close()
            if len(receptor_all_chains) != 1:
                continue
            receptor_all_chains = list(receptor_all_chains)
            if receptor_all_chains[0] == ' ':
                receptor_chain = '_'
            else:
                receptor_chain = receptor_all_chains[0]
        else:
            receptor_pdb, receptor_chain = receptor.split('_')
            receptor_pdb = os.path.join(PDB_DATA_DIR, receptor_pdb[1:-1].lower(), 'pdb%s.ent.gz' % receptor_pdb.lower())
            
            
        if len(ligand) == 32:
            if os.path.isfile(os.path.join(non_cached_dir, 'modbase', 'models', 'hash', ligand + '.pdb')):
                ligand_pdb = os.path.join(non_cached_dir, 'modbase', 'models', 'hash', ligand + '.pdb')
            else:
                ligand_pdb =  os.path.join(MODBASE_CACHE, 'models', 'hash', ligand + '.pdb')
                
            ligand_all_chains = set()
            infile_ligand = open(ligand_pdb, 'r')
            for line in infile_ligand:
                if line.split()[0] == 'ATOM':
                    ligand_all_chains.add(line[21])
            infile_ligand.close()
            if len(ligand_all_chains) != 1:
                continue
            ligand_all_chains = list(ligand_all_chains)
            if ligand_all_chains[0] == ' ':
                ligand_chain = '_'
            else:
                ligand_chain = ligand_all_chains[0]
        else:
            ligand_pdb, ligand_chain = ligand.split('_')
            ligand_pdb = os.path.join(PDB_DATA_DIR, ligand_pdb[1:-1].lower(), 'pdb%s.ent.gz' % ligand_pdb.lower())
            
        if receptor_chain != '_':
            check_receptor = os.path.splitext(os.path.basename(receptor_pdb))[0].upper()+'_'+receptor_chain
        else:
            check_receptor = os.path.splitext(os.path.basename(receptor_pdb))[0].upper()
        if ligand_chain != '_':
            check_ligand = os.path.splitext(os.path.basename(ligand_pdb))[0].upper()+'_'+ligand_chain
        else:
            check_ligand = os.path.splitext(os.path.basename(ligand_pdb))[0].upper()
            
        if os.path.exists(os.path.join(ZDOCK_CACHE, 'mixed_docked_models', '%s--%s--ZDOCK.out' % (check_receptor, check_ligand))):
            mixed_receptor_ligand.add(os.path.join(ZDOCK_CACHE, 'mixed_docked_models', '%s--%s--ZDOCK*.pdb' % (check_receptor, check_ligand)))
            continue
            
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
        receptor_chain_list.append(receptor_chain)
        ligand_chain_list.append(ligand_chain)
        
        mixed_receptor_ligand.add(os.path.join(mixed_docked_model_dir, '%s--%s--ZDOCK*.pdb' % (check_receptor, check_ligand)))
        flag = True
        
    if flag:
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        my_pool = Pool(10) # PARAM
        my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list, [mixed_docked_model_dir] * len(receptor_pdb_list)))
        my_pool.close()
        my_pool.join()
    
    # Step 2.4: Write docking sets as temporary files.
    with open(os.path.join(non_cached_dir, 'zdock', 'docked_models', 'pdb_docked_models.txt'), 'w') as f:
        for e in pdb_receptor_ligand:
            f.write(e + '\n')
            
    with open(os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mb_docked_models.txt'), 'w') as f:
        for e in modbase_receptor_ligand:
            f.write(e + '\n')
        f.write('\nmodbase models\n')
        f.write(os.path.join(non_cached_dir, 'modbase', 'parsed_files', 'select_modbase_models.txt')+'\n')
        f.write(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt')+'\n')
        f.write(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt')+'\n')
            
    with open(os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mixed_docked_models.txt'), 'w') as f:
        for e in mixed_receptor_ligand:
            f.write(e + '\n')
        f.write('\nmodbase models\n')
        f.write(os.path.join(non_cached_dir, 'modbase', 'parsed_files', 'select_modbase_models.txt')+'\n')
        f.write(os.path.join(MODBASE_CACHE, 'parsed_files', 'm3D_modbase_models.txt')+'\n')
        f.write(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt')+'\n')
    
    # Step 3: Calculate ZDOCK features for each docked model.
    # Step 3.1: Calculate for PDB docked models.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s pdb -d %r' % (os.path.join(non_cached_dir, 'zdock', 'docked_models', 'pdb_docked_models.txt'), os.path.join(non_cached_dir, 'zdock', 'docked_models', 'dist3d_pdb_docking.txt'), json.dumps(seq_dict))) # PARAMS
    
    # Step 3.2: Calculate for ModBase docked models first.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s modbase -d %r' % (os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mb_docked_models.txt'), os.path.join(non_cached_dir, 'zdock', 'docked_models', 'dist3d_modbase_docking.txt'), json.dumps(seq_dict))) # PARAMS
    
    # Step 3.3: Calculate for mixed docked models.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s mixed -d %r' % (os.path.join(non_cached_dir, 'zdock', 'docked_models', 'mixed_docked_models.txt'), os.path.join(non_cached_dir, 'zdock', 'docked_models', 'dist3d_mixed_docking.txt'), json.dumps(seq_dict))) # PARAMS
    
    # Step 4: Aggregate ZDOCK feature.
    interaction2dist3d = defaultdict(lambda: defaultdict(lambda: ([], [])))  # (p1, p2) -> (subA, subB) -> ([dsasas1...], [dsasas2...])
    pdb2uniprots = defaultdict(set)                                          # Store all uniprots seen in each PDB
    dist3d_lists = [parse_dictionary_list(os.path.join(non_cached_dir, 'zdock', 'docked_models', f)) for f in ['dist3d_pdb_docking.txt', 'dist3d_mb_docking.txt', 'dist3d_mixed_docking.txt'] if os.path.exists(os.path.join(non_cached_dir, 'zdock', 'docked_models', f))]
    for e in sum(dist3d_lists, []):
        interaction = (e['UniProtA'], e['UniProtB'])
        dock_pair = (e['SubunitA'], e['SubunitB'])
        zdock_score = float(e['ZDOCK_Score'])
        if interaction not in int_list:
            continue
        if interaction in interaction2dist3d and dock_pair not in interaction2dist3d[interaction]:
            continue
        if zdock_score < 0: # PARAMS, docking_score_cutoff
            continue
        dist3ds1 = np.array([float(r) if r != 'nan' else np.nan for r in e['UniProtA_dist3d'].split(';')])
        dist3ds2 = np.array([float(r) if r != 'nan' else np.nan for r in e['UniProtB_dist3d'].split(';')])
        if all((dist3ds1 == 0.0) | np.isnan(dist3ds1)) or all((dist3ds2 == 0.0) | np.isnan(dist3ds2)):
            continue
        if e['UniProtA'] == e['UniProtB']:  # Homodimers have the same features for both proteins, save minimum dist3d for either subunit.
            both_dist3ds = [dist3ds1, dist3ds2]
            nan_mask = np.ma.masked_array(both_dist3ds, np.isnan(both_dist3ds))
            dist3d_min = np.min(nan_mask, axis=0)
            dist3d_min = np.array([dist3d_min.data[r] if dist3d_min.mask[r] == False else np.nan for r in range(len(dist3d_min.data))])
            interaction2dist3d[interaction][dock_pair][0].append(dist3d_min)
            interaction2dist3d[interaction][dock_pair][1].append(dist3d_min)
        else:
            interaction2dist3d[interaction][dock_pair][0].append(dist3ds1)
            interaction2dist3d[interaction][dock_pair][1].append(dist3ds2)
            
    zdock_feat_dict = {'top1': {}, 'max': {}, 'mean': {}, 'min': {}}
    for i in interaction2dist3d:
        id1_dist3ds, id2_dist3ds = [], []
        for docki, dock_pair in enumerate(interaction2dist3d[i]):
            if docki == 0:
                zdock_feat_dict['top1'][i] = (interaction2dist3d[i][dock_pair][0][0], interaction2dist3d[i][dock_pair][1][0])
            id1_dist3ds += interaction2dist3d[i][dock_pair][0]
            id2_dist3ds += interaction2dist3d[i][dock_pair][1]

        mdat = np.ma.masked_array(id1_dist3ds, np.isnan(id1_dist3ds))
        id1_mean = np.mean(mdat, axis=0)
        id1_mean = [id1_mean.data[r] if id1_mean.mask[r]==False else np.nan for r in range(len(id1_mean.data))]
        id1_max = np.max(mdat, axis=0)
        id1_max = [id1_max.data[r] if id1_max.mask[r]==False else np.nan for r in range(len(id1_max.data))]
        id1_min = np.min(mdat, axis=0)
        id1_min = [id1_min.data[r] if id1_min.mask[r]==False else np.nan for r in range(len(id1_min.data))]
        mdat = np.ma.masked_array(id2_dist3ds, np.isnan(id2_dist3ds))
        id2_mean = np.mean(mdat, axis=0)
        id2_mean = [id2_mean.data[r] if id2_mean.mask[r]==False else np.nan for r in range(len(id2_mean.data))]
        id2_max = np.max(mdat, axis=0)
        id2_max = [id2_max.data[r] if id2_max.mask[r]==False else np.nan for r in range(len(id2_max.data))]
        id2_min = np.min(mdat, axis=0)
        id2_min = [id2_min.data[r] if id2_min.mask[r]==False else np.nan for r in range(len(id2_min.data))]
        zdock_feat_dict['max'][i] = (np.array(id1_max), np.array(id2_max))
        zdock_feat_dict['mean'][i] = (np.array(id1_mean), np.array(id2_mean))
        zdock_feat_dict['min'][i] = (np.array(id1_min), np.array(id2_min))
        
    return zdock_feat_dict

def calculate_raptorX(seq_dict, non_cached_dir):
    """
    Calculate RaptorX features.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A dictionary of RaptorX features mapping id -> feature -> raptorx_array.
    
    """
    raptorx_dir_dict = {}
    # Step 1: Run RaptorX for all proteins.
    cur_dir = os.getcwd()
    for prot in seq_dict:
        flag = False
        cached_raptorx_file = os.path.join(RAPTORX_CACHE, prot)
        if os.path.exists(cached_raptorx_file):
            if os.path.exists(os.path.join(cached_raptorx_file, '%s.ss3' % prot)) and os.stat(os.path.join(cached_raptorx_file, '%s.ss3' % prot)).st_size != 0 and os.path.exists(os.path.join(cached_raptorx_file, '%s.acc' % prot)) and os.stat(os.path.join(cached_raptorx_file, '%s.acc' % prot)).st_size != 0:
                ss3_df = pd.read_csv(os.path.join(cached_raptorx_file, '%s.ss3' % prot), skiprows=2, sep='\s+', header=None)
                ss3_length = ss3_df[3].values.shape[0]
                
                acc_df = pd.read_csv(os.path.join(cached_raptorx_file, '%s.acc' % prot), skiprows=3, sep='\s+', header=None)
                acc_length = acc_df[3].values.shape[0]

                prot_length = len(seq_dict[prot])
                if ss3_length == prot_length and acc_length == prot_length:
                    raptorx_dir_dict[prot] = cached_raptorx_file
                    flag = True
                
        if not flag:
            raptorx_dir = os.path.join(non_cached_dir, 'raptorx')
            if not os.path.exists(raptorx_dir):
                os.system('mkdir -p '+raptorx_dir)
                
            os.chdir(RAPTORX)
            generate_fasta(prot, seq_dict[prot], raptorx_dir)
            os.system(os.path.join('./oneline_command.sh' + ' ' + os.path.join(raptorx_dir, prot + '.fasta') + ' ./tmp 2 1')) # PARAMS
            os.system('mv ./tmp/' + prot + ' ' + raptorx_dir)
            raptorx_dir_dict[prot] = os.path.join(raptorx_dir, prot)
            os.chdir(cur_dir)
    
    # Step 2: Aggregate RaptorX features.
    raptorx_dict = {}
    for feat in ['SS_H_prob', 'SS_E_prob', 'SS_C_prob', 'ACC_B_prob', 'ACC_M_prob', 'ACC_E_prob']:
        raptorx_dict[feat] = {}
    for prot in seq_dict:
        prot_length = len(seq_dict[prot])
        raptorx_ss3_file = os.path.join(raptorx_dir_dict[prot], '%s.ss3' % prot)
        raptorx_acc_file = os.path.join(raptorx_dir_dict[prot], '%s.acc' % prot)
        
        nan_array = np.empty((prot_length))
        nan_array[:] = np.nan
        if os.path.exists(raptorx_ss3_file) and os.stat(raptorx_ss3_file).st_size != 0:
            ss3_df = pd.read_csv(raptorx_ss3_file, skiprows=2, sep='\s+', header=None)
            ss3_length = ss3_df[3].values.shape[0]
            if ss3_length == prot_length:
                raptorx_dict['SS_H_prob'][prot] = ss3_df[3].values
                raptorx_dict['SS_E_prob'][prot] = ss3_df[4].values
                raptorx_dict['SS_C_prob'][prot] = ss3_df[5].values
            else:
                raptorx_dict['SS_H_prob'][prot] = nan_array
                raptorx_dict['SS_E_prob'][prot] = nan_array
                raptorx_dict['SS_C_prob'][prot] = nan_array
        else:
            raptorx_dict['SS_H_prob'][prot] = nan_array
            raptorx_dict['SS_E_prob'][prot] = nan_array
            raptorx_dict['SS_C_prob'][prot] = nan_array
        if os.path.exists(raptorx_acc_file) and os.stat(raptorx_acc_file).st_size != 0:
            acc_df = pd.read_csv(raptorx_acc_file, skiprows=3, sep='\s+', header=None)
            acc_length = acc_df[3].values.shape[0]
            if acc_length == prot_length:
                raptorx_dict['ACC_B_prob'][prot] = acc_df[3].values
                raptorx_dict['ACC_M_prob'][prot] = acc_df[4].values
                raptorx_dict['ACC_E_prob'][prot] = acc_df[5].values
            else:
                raptorx_dict['ACC_B_prob'][prot] = nan_array
                raptorx_dict['ACC_M_prob'][prot] = nan_array
                raptorx_dict['ACC_E_prob'][prot] = nan_array
        else:
            raptorx_dict['ACC_B_prob'][prot] = nan_array
            raptorx_dict['ACC_M_prob'][prot] = nan_array
            raptorx_dict['ACC_E_prob'][prot] = nan_array
    return raptorx_dict

def calculate_pair_potential(int_list, seq_dict, non_cached_dir):
    """
    Calculate pair potential features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        non_cached_dir (str): Output directory for each specific feature.
        
    Returns:
        A tuple two dictionaries. The first maps (id1, id2) -> (pp_1, pp_2) and the second maps (id1, id2) ->
        (pp_normed_1, pp_normed_2).
    
    """
    # Load pair potential background matrices
    g_homo = np.load(pkg_resources.resource_filename(__name__, 'data/G_homo.npy'))
    g_hetero = np.load(pkg_resources.resource_filename(__name__, 'data/G_hetero.npy'))
    
    pp_dict, pp_norm_dict = {}, {}
    pp = PairPotential()
    for id1, id2 in int_list:
        if id1 == id2:
            flag = False
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))):
                try:
                    df = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))

                    if df.shape[0] == len(seq_dict[id1]):
                        pp_dict[(id1, id2)] = (df['pair_potential'].values, df['pair_potential'].values)
                        pp_norm_dict[(id1, id2)] = (df['pair_potential_norm'].values, df['pair_potential_norm'].values)
                        flag = True
                except:
                    pass
                    
            if not flag:
                pp_array = pp.get_feature(seq_dict[id1], seq_dict[id2], g_homo)
                pp_normed = normalize_feat(pp_array)
                data = {'pair_potential': pp_array, 'pair_potential_norm': pp_normed}
                if not os.path.exists(os.path.join(non_cached_dir, 'pp')):
                    os.system('mkdir -p '+os.path.join(non_cached_dir, 'pp'))
                pd.DataFrame(data).to_pickle(os.path.join(non_cached_dir, 'pp', '%s_%s_0.pkl' % (id1, id2)))
                pp_dict[(id1, id2)] = (pp_array, pp_array)
                pp_norm_dict[(id1, id2)] = (pp_normed, pp_normed)
        else:
            flag = False
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))) and os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2))):
                try:
                    df1 = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))
                    df2 = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2)))

                    if df1.shape[0] == len(seq_dict[id1]) and df2.shape[0] == len(seq_dict[id2]):
                        pp_dict[(id1, id2)] = (df1['pair_potential'].values, df2['pair_potential'].values)
                        pp_norm_dict[(id1, id2)] = (df1['pair_potential_norm'].values, df2['pair_potential_norm'].values)
                        flag = True
                except:
                    pass
                    
            if not flag:
                # For protein 1
                pp_array1 = pp.get_feature(seq_dict[id1], seq_dict[id2], g_hetero)
                pp_normed1 = normalize_feat(pp_array1)
                data1 = {'pair_potential': pp_array1, 'pair_potential_norm': pp_normed1}
                if not os.path.exists(os.path.join(non_cached_dir, 'pair_potential')):
                    os.system('mkdir -p '+os.path.join(non_cached_dir, 'pair_potential'))
                pd.DataFrame(data1).to_pickle(os.path.join(non_cached_dir, 'pair_potential', '%s_%s_0.pkl' % (id1, id2)))
                # For protein 2
                pp_array2 = pp.get_feature(seq_dict[id2], seq_dict[id1], g_hetero)
                pp_normed2 = normalize_feat(pp_array2)
                data2 = {'pair_potential': pp_array2, 'pair_potential_norm': pp_normed2}
                pd.DataFrame(data2).to_pickle(os.path.join(non_cached_dir, 'pair_potential', '%s_%s_1.pkl' % (id1, id2)))
                
                pp_dict[(id1, id2)] = (pp_array1, pp_array2)
                pp_norm_dict[(id1, id2)] = (pp_normed1, pp_normed2)
                
    return pp_dict, pp_norm_dict
