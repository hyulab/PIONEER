import os
import glob
import subprocess
from shutil import rmtree
from tempfile import mkdtemp
from collections import defaultdict
from .config import NACCESS
from .utils import natural_keys, open_pdb


def naccess(pdb_file):
    """
    Run NACCESS and return the results.
    
    Args:
        pdb_file (str): File to run NACCESS for.
    
    Returns:
        A string where each line contains the chain, residue number, amino acid and SASA for one residue in
        the structure specified.
    
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(pdb_file))
    raw_naccess_output = []
    if(os.path.exists(os.path.splitext(pdb_file)[0]+'.rsa')):
        os.remove(os.path.splitext(pdb_file)[0]+'.rsa')
    if(os.path.exists(os.path.splitext(pdb_file)[0]+'.asa')):
        os.remove(os.path.splitext(pdb_file)[0]+'.asa')
    _, _ = subprocess.Popen([NACCESS, pdb_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    try:
        raw_naccess_output += open(os.path.splitext(pdb_file)[0]+'.rsa').readlines()
        if(os.stat(os.path.splitext(pdb_file)[0]+'.rsa').st_size == 0):
            os.chdir(cwd)
            return raw_naccess_output
#             raise ValueError('ERROR: Naccess .rsa file is empty. We suspect this is an edge case where Naccess cannot calculate ASA for extremely large chains. The following command was attempted: %s %s' %(NACCESS, pdb_file))
    except:
        os.chdir(cwd)
        return raw_naccess_output
#         raise IOError('ERROR: Naccess .rsa file was not written. The following command was attempted: %s %s' %(NACCESS, pdb_file))
    os.chdir(cwd)
    return raw_naccess_output
    

def srescalc(structure, chain=None, comp='Isolation', uSASA=15):
    """
    Calculate SASA for an input structure. Originally written by Michael Meyer.
    
    Args:
        structure (str): File name of input PDB file (.pdb or .ent), or gzipped pdb file (.gz), or 4-letter pdb ID. 
            If no argument given, reads structure on STDIN.
        chain (str): Chain to calculate. If not given, all chains will be calculated.
        comp (str): Calculate surface residues on intact complex (not with each chain in isolation). 
        uSASA (int): Minimum percent solvent accessibility of unbound residues to be considered a surface residue. 
            Integer [0-100].
        
    Returns:
        A string where each line contains the chain, residue number, amino acid and SASA for one residue.
    
    """
    pdbinfile = open_pdb(structure)
    pdbatomdict = {}
    curchain = ''
    
    # NEW
    for line in pdbinfile:
        if type(line) == bytes:
            line = line.decode('utf-8')
        if line is not None and line[:4] == "ATOM":
            if curchain != line[21]:
                curchain = line[21]
                pdbatomdict[curchain] = [line]
            else:
                pdbatomdict[curchain].append(line)
    pdbinfile.close()
    
    # CHECK THAT SELECTED CHAINS EXIST IN GIVEN PDB FILE
    user_chains = set()
    if comp != 'Isolation':
        user_chains = set(comp.split('/'))
    elif chain != None:
        user_chains = set([chain])
    if len(user_chains) > 0 and user_chains.intersection(set(pdbatomdict.keys())) != user_chains:
        return ''
#         raise ValueError('Error: No chains in file match filter. Remember, PDB chains are case-sensitive.\nAvailable Chains: %s\nSelected Chains: %s' %(', '.join(pdbatomdict.keys()), ', '.join(list(user_chains))))
    
    # CREATE SCRATCH SPACE FOR WRITING INTERMEDIATE FILES
    scratchDir = mkdtemp()
    tempfilename = os.path.join(scratchDir, 'temp.txt')
    
    pdbchainlist = sorted(pdbatomdict.keys())
    temp_pdb_files = []
    if comp == 'Isolation':
        for h in pdbchainlist:
            if chain is not None and h != chain:
                continue
            if h == ' ':
                temp_file_name = '_.pdb'
            else:
                temp_file_name = h + '.pdb'
            temp_pdb_files.append(os.path.join(scratchDir, temp_file_name))
            with open(temp_pdb_files[-1], 'w') as tempoutfile:
                for line in pdbatomdict[h]:
                    tempoutfile.write(line)
    else:
        if comp == 'ALL':
            complex_chains = pdbchainlist
        else:
            complex_chains = comp.split('/')
        temp_pdb_files.append(os.path.join(scratchDir,'complex.pdb'))
        with open(temp_pdb_files[-1], 'w') as tempoutfile:
            for h in complex_chains:
                for line in pdbatomdict[h]:
                    tempoutfile.write(line)
    
    # RUN NACCESS
    naccess_output = []
    for pdb_file in temp_pdb_files:
        naccess_output += naccess(pdb_file)
    
    # PARSE NACCESS OUTPUT
    asadict = defaultdict(list)
    for line in naccess_output:
        if line[:3] != 'RES':
            continue
        aa = line[3:7].strip()
        chain = line[7:9].strip()
        residue_index = line[9:13].strip()
        relative_perc_accessible = float(line[22:28])
        if relative_perc_accessible < uSASA:
            continue
        asadict[chain].append((residue_index, aa, relative_perc_accessible))
    
    # FORMAT AND PRINT OUTPUT
    output = ''
    for k in sorted(asadict.keys()):
        indices = sorted([res[0] for res in asadict[k]], key=natural_keys)
        for res in asadict[k]:
            output += '%s\t%s\t%s\t%s\n' %(k, res[0], res[1], res[2])

    # CLEANUP
    rmtree(scratchDir)
    return output


def calculate_SASA(hash_file, uniprot_length, header_info, out_dir):
    """
    Calculate SASA for a ModBase model.
    
    Args:
        hash_file (str): Hash file of the ModBase model.
        uniprot_length (int): Length of the corresponding UniProt.
        header_info (str): Information about the ModBase model as documented in the summary file.
        out_dir (str): Path to the directory to store the output.
        
    Returns:
        None.
    
    """
    if os.path.exists(os.path.join(out_dir, header_info.split('\t')[-1] + '.txt')): # Result already exists
        return
    out = srescalc(hash_file, uSASA=-1)
    out = [x.replace('\n', '').split('\t') for x in out.split('\n')[:-1]]
    try:
        uniprotSASA = dict([(int(q[1]), q[3]) for q in out]) # map UniProt residues to SASA values
    except:
        print('Error calculating SASA for %s.' % hash_file)
        return
    SASAs = [str(uniprotSASA[r]) if r in uniprotSASA else 'NaN' for r in range(1, uniprot_length + 1)]
    with open(os.path.join(out_dir, header_info.split('\t')[-1] + '.txt'), 'w') as output_f:
        output_f.write('\t'.join([header_info] + [';'.join(SASAs)]) + '\n')
    
    
def gather_SASA(sasa_dir, output_file):
    """
    Gather calculated SASA information and append it to the output file.
    
    Args:
        sasa_dir (str): Directory exclusively storing SASA output (from `calculate_SASA`).
        output_file (str): Path to the SASA information file (`SASA_modbase_other.txt`).
        
    Returns:
        None.
    
    """
    columns = ['uniprot', 'template_length', 'target_length', 'template_pdb', 'template_chain', 'target_begin', 'target_end', 'sequence_identity', 'model_score', 
               'modpipe_quality_score', 'zDOPE', 'eVALUE', 'modbase_modelID', 'SASA']
    with open(output_file, 'a') as out_f:
        for _, file in enumerate(glob.glob(os.path.join(sasa_dir, '*.txt'))):
            out_list = []
            with open(file) as f:
                for line in f:
                    for col, val in zip(columns, line.strip().split('\t')):
                        if col != 'template_chain':
                            out_list.append(val)
            out_f.write('\t'.join(out_list) + '\n')
