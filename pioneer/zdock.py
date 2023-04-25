import os
import re
import sys
import glob
import subprocess
import numpy as np
from tempfile import mkdtemp
from collections import defaultdict
from shutil import rmtree, copyfile
from multiprocessing import Pool, Manager
from .utils import *
from .config import *
from .srescalc import naccess


def zdock(receptor, ligand, rc='A', lc='A', outdir='.', num_models=2000, top_models=10, seed=-1, fix_receptor=True, 
          fix_modbase=False, name=None, gzip=False, verbose=False):
    """
    Perform ZDOCK for two structural models. Originally written by Michael. Modified by Charles Liang.
    
    Args:
        receptor (str): File name of receptor PDB file (.pdb or .ent), or gzipped pdb file (.gz),
            or 4-letter pdb ID.
        ligand (str): File name of ligand PDB file (.pdb or .ent), or gzipped pdb file (.gz),
            or 4-letter pdb ID.
        rc (str): Chain of receptor file to dock. _ for no chain ID (ModBase).
        lc (str): Chain of ligand file to dock. _ for no chain ID (ModBase).
        outdir (str): Path to the directory to store output files.
        num_models (int): Number of ZDOCK models to calculate.
        top_models (int): Number of top ZDOCK models to produce as PDB files.
        seed (int): The seed for the randomization of the starting PDBs (default to no randomization).
        fix_receptor (bool): If True, fix the receptor, preventing its rotation and switching with 
            ligand during execution.
        fix_modbase (bool): If True, remove fields 73-80 in model file (tend to occur in modbase files, 
            messing up docking). Replace with single element ID in field 77.
        name (str): Prefix attached to all output files. (default: a prefix will be created from 
            the given subunits and chains).
        gzip (bool): If True, gzip docked pdb structures (.pdb.gz).
        verbose (bool): If True, show steps and warnings.
        
    Returns:
        None.
    
    """
    # Isolate chains.
    scratch_dir = mkdtemp()
    if verbose:
        print('writing preliminary data to %s...' %(scratchDir))
    if rc == '_': 
        rc = ' '
    if lc == '_': 
        lc = ' '
    
    receptor_basename = os.path.splitext(os.path.basename(receptor))[0].upper()
    if rc != ' ': receptor_basename += '_' + rc
    cleaned_receptor = os.path.join(scratch_dir, '%s.R' %(receptor_basename))
    extract_atoms(receptor, cleaned_receptor, chain_dict = {rc:set()}, chain_map = {rc: 'A'}, fix_modbase=fix_modbase)
    
    ligand_basename = os.path.splitext(os.path.basename(ligand))[0].upper()
    if lc != ' ': ligand_basename += '_' + lc
    cleaned_ligand = os.path.join(scratch_dir, '%s.L' %(ligand_basename))
    extract_atoms(ligand, cleaned_ligand, chain_dict = {lc:set()}, chain_map = {lc: 'B'}, fix_modbase=fix_modbase)
    
    if name is None:
        run_id = '%s--%s--ZDOCK' % (receptor_basename, ligand_basename)
    else:
        run_id = name
    
    # Run ZDOCK.
    cur_dir = os.getcwd()
    os.chdir(scratch_dir)
    zdock_output_file = os.path.join(scratch_dir, 'zdock.out')
    zdock_params = [ZDOCK, '-R', cleaned_receptor, '-L', cleaned_ligand, '-N', '%i' % num_models]
    if fix_receptor:
        zdock_params += ['-F']
    if seed != -1:
        zdock_params += ['-S', '%i' %(seed)]
    zdock_params += ['-o',  zdock_output_file]
    if verbose:
        print(' '.join(zdock_params))
    out, err = subprocess.Popen(zdock_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    # Process output file if it exists.
    if os.path.exists(zdock_output_file):
        
        # Model creation.
        copyfile(CREATE_PL, os.path.join(scratch_dir, 'create.pl'))
        os.system('chmod 777 %s' %(os.path.join(scratch_dir, 'create.pl')))
        copyfile(CREATE_LIG, os.path.join(scratch_dir, 'create_lig'))
        os.system('chmod 777 %s' %(os.path.join(scratch_dir, 'create_lig')))
        top_models_file = zdock_output_file + '.top'
        os.system('head -%i %s > %s' % (4 + top_models, zdock_output_file, top_models_file))
        os.system('./create.pl %s' % top_models_file)
        os.chdir(cur_dir)
        
        new_zdock_outfile = os.path.join(outdir, '%s.out' %(run_id))
        copyfile(zdock_output_file, new_zdock_outfile)

        # Remove tmp paths from receptor and ligand file names in zdock output file.
        replace_command = "grep -rl '%s/' %s | xargs sed -i 's/%s\///g'" % (scratch_dir, new_zdock_outfile, scratch_dir.replace('/', '\/'))
        os.system(replace_command)
        copyfile(cleaned_receptor, os.path.join(outdir, os.path.basename(cleaned_receptor)))
        copyfile(cleaned_ligand, os.path.join(outdir, os.path.basename(cleaned_ligand)))
        complex_files = glob.glob(os.path.join(scratch_dir, '*.pdb'))
        if gzip:
            for f in complex_files:
                os.system('gzip %s' %(f))
            complex_files = glob.glob(os.path.join(scratch_dir, '*.pdb.gz'))
        for f in complex_files:
            num = os.path.basename(f).split('.')[1]
            if len(num) == 1:
                num = '0' + num
            if gzip:
                new_file = os.path.join(outdir, '%s-%s.pdb.gz' %(run_id, num))
            else:
                new_file = os.path.join(outdir, '%s-%s.pdb' %(run_id, num))
            copyfile(f, new_file)
    
    # Clean up temporary directory.
    rmtree(scratch_dir)
    
    
def chaindistcalc(structure, c1, c2, nres=10):
    """
    Calculate contacting pairs of residues between two chains in a PDB file using simple 3D distances, 
    or 3D distances only between interface residues. Originally written by Michael J Meyer, and modified
    by Charles Liang.
    
    Args:
        structure (str): File name of input PDB file (.pdb or .ent), or gzipped pdb file (.gz), 
            or 4-letter pdb ID.
        c1 (str): First of chain pair for which to calculate contact residues.
        c2 (str): Second of chain pair for which to calculate contact residues.
        nres (int): Number of shortest pairwise residue distances to average to find distance of 
            each residue to other chain.
            
    Returns:
        A string where each row contains one residue in one chain and its average distance to `nres` closest
        residues in the other chain.
    
    """
    # Parse PDB file.
    pdbinfile = open_pdb(structure)
    pdb_map = {}
    use_atoms = set(['CA'])
    for l in pdbinfile:
        record_name = l[:6].strip()
        atom_name = l[12:16].strip()
        if record_name != 'ATOM':
            continue
        if atom_name not in use_atoms:
            continue
        chain_id, res_name, res_seq, x, y, z = l[21], l[17:20].strip(), l[22:27].strip(), float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip())
        if chain_id not in pdb_map:
            pdb_map[chain_id] = {}
        if res_seq not in pdb_map[chain_id]:
            pdb_map[chain_id][res_seq] = []
        pdb_map[chain_id][res_seq].append({'atom': atom_name, 'residue': res_name, 'coords': (x,y,z), 'full_record': l})
    
    # Check chains.
    available_chains = sorted(pdb_map.keys())
    if c1 is None or c2 is None:
        if len(available_chains) != 2:
            print('No chains specified for structure with > 2 available chains: %s' %(', '.join(available_chains)))
            return
        else:
            c1 = available_chains[0]
            c2 = available_chains[1]
    elif c1 not in available_chains or c2 not in available_chains:
        print('One or both chains %s and %s not found in file. Available chains: %s' %(c1, c2, ', '.join(available_chains)))
        return
    
    # Calculate CA distances.
    res_dists = defaultdict(list)   # Map residue pairs to the distance between them
    candidate_res1 = sorted(list(pdb_map[c1]), key=natural_keys)
    candidate_res2 = sorted(list(pdb_map[c2]), key=natural_keys)
    for i1, r1 in enumerate(candidate_res1):
        for i2, r2 in enumerate(candidate_res2):
            res_pair = (r1, r2)
            a1 = pdb_map[c1][r1][0]
            a2 = pdb_map[c2][r2][0]
            atomdist = ((a1['coords'][0] - a2['coords'][0]) ** 2 + (a1['coords'][1] - a2['coords'][1]) ** 2 + (a1['coords'][2] - a2['coords'][2]) ** 2) ** 0.5
            res_dists[(c1, r1, a1['residue'])].append(atomdist)
            res_dists[(c2, r2, a2['residue'])].append(atomdist)
    
    # Format output.
    outstr = ''
    for chain, res_num, res_name in sorted(res_dists.keys(), key=lambda l: natural_keys(' '.join(l))):
        cur_res_dists = sorted(res_dists[(chain, res_num, res_name)])
        avg_distance = np.mean(cur_res_dists[:nres])
        outstr += '\t'.join([chain, res_num, res_name, '%.4f' % avg_distance]) + '\n'
    return outstr
        
        
def irescalc(structure, c1, c2, uSASA=15, dSASA=1.0, outfmt='list'):
    """
    Calculate interface residues between specified chains in PDB files. If only two chains present 
    in structure, no need to specify. Originally written by Michael J Meyer, and modified by Shayne
    Wierbowski and Charles Liang.
    
    Args:
        structure (str): File name of input PDB file (.pdb or .ent), or gzipped pdb file (.gz), 
            or 4-letter pdb ID.
        c1 (str): First of chain pair for which to calculate interface residues.
        c2 (str): Second of chain pair for which to calculate interface residues.
        uSASA (str): Minimum percent solvent accessibility of an unbound residue to be considered 
            a surface residue. Eliminate criteria with value of 0.
        dSASA (float): Minimum change in solvent accessible surface area (in Angstroms squared) for 
            a surface residue to be considered an interface residue.
        outfmt (str): Format output options, 'list' or 'residue_stats'.
            
    Returns:
        If `outfmt` is 'list', return a string specifying interface residues identified from the two
        chains. Otherwise, return a string where each row indicates the dSASA value of one residue in
        one chain.
    
    """
    # Parse PDB file.
    pdbinfile = open_pdb(structure)
    pdbatomdict = {}
    curchain = ''
    for line in pdbinfile:
        if line is not None and line[:4] == 'ATOM':
            if curchain != line[21]:
                curchain = line[21]
                pdbatomdict[curchain] = [line]
            else:
                pdbatomdict[curchain].append(line)
    pdbinfile.close()
    
    # CREATE SCRATCH SPACE FOR WRITING INTERMEDIATE FILES
    scratch_dir = mkdtemp()
    tempfilename = os.path.join(scratch_dir, 'temp')
    
    # Write PDB files, and then generate RSA, ASA and LOG files by using Naccess to calculate solvent accessibilities,
    # and finally use RSA files to calculate interface residues.
    asadict = {}
    pdbchainlist = sorted(pdbatomdict.keys())
    
    # Pick pair of chains to calc ires based on user input or default to only two chains in structure.
    if c1 is None or c2 is None:  #chains not specified
        if len(pdbchainlist) == 2:
            i = 0; j = 1
        else:
            print('No chains specified for structure with > 2 available chains.')
            return
    else: #specific chains specified
        try:
            i = pdbchainlist.index(c1)
            j = pdbchainlist.index(c2)
        except:
            print('One or both chains %s and %s not found in file' % (c1, c2))
            return
    
    for chain in pdbchainlist:
        # EDIT: Made By Shayne (July 12, 2018)
        # I have modified this loop to only calculate ASA for the two input chains
        if(not pdbchainlist.index(chain) in [i, j]):
            continue
        asadict[chain] = {}
        temp_pdb_file = tempfilename + '.pdb'
        
        # Write unbound form of chain for naccess calculations
        tempoutfile = open(temp_pdb_file,'w')
        for line in pdbatomdict[chain]:
            tempoutfile.write(line)
        tempoutfile.close()

        # Run NACCESS
        naccess_output = naccess(temp_pdb_file)
        for res in naccess_output:
            if res['all_atoms_rel'] >= uSASA:      # relative percent SASA >= 15% indicates surface residue
                residue_key = (res['res'], res['chain'], res['res_num'])
                asadict[chain][residue_key] = res['all_atoms_abs']   #save absolute SASA to compare with bound form
    
    # Calculalate interface residues for the given chain pair
    asadict[pdbchainlist[i] + pdbchainlist[j]] = {}

    # Write temporary file with only those two chains for processing with naccess
    temp_pdb_file = tempfilename + '.pdb'
    tempoutfile = open(temp_pdb_file, 'w')
    for line in pdbatomdict[pdbchainlist[i]]:
        tempoutfile.write(line)
    for line in pdbatomdict[pdbchainlist[j]]:
        tempoutfile.write(line)
    tempoutfile.close()
    
    # Run NACCESS
    naccess_output = naccess(temp_pdb_file)
    intreslist = {}
    intreslist[pdbchainlist[i]] = []
    intreslist[pdbchainlist[j]] = []
    for res in naccess_output:
        residue_key = (res['res'], res['chain'], res['res_num'])
        chain = res['chain']
        if residue_key not in asadict[chain]:
            continue
        res_dSASA = abs(asadict[chain][residue_key] - res['all_atoms_abs'])
        if res_dSASA >= dSASA:    # Change in solvent accessible surface area >= 1 A squared
            intreslist[chain].append((chain, res['res_num'], res['res'], res_dSASA))
    
    # Cleanup
    rmtree(scratch_dir)
    
    outstr = ''
    if outfmt == 'list':
        outstr += ','.join([q[1] for q in intreslist[pdbchainlist[i]]]) + '\n'
        outstr += ','.join([q[1] for q in intreslist[pdbchainlist[j]]]) + '\n'
    elif outfmt == 'residue_stats':
        for res in intreslist[pdbchainlist[i]]:
            outstr += '%s\t%s\t%s\t%s\n' % tuple(res)
        for res in intreslist[pdbchainlist[j]]:
            outstr += '%s\t%s\t%s\t%s\n' % tuple(res)
    return outstr
