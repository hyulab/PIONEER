import os
from .config import PSIBLAST, UNIPROT_ALL_DB, CLUSTALO, UNIPROT_SEQS_PATH


def generate_fasta(prot_id, sequence, out_dir):
    with open(os.path.join(out_dir, prot_id + '.fasta'), 'w') as f:
        f.write('>' + prot_id + '\n')
        f.write(sequence + '\n')
    return os.path.join(out_dir, prot_id + '.fasta')

def generate_single_msa(prot_id, sequence, out_dir):
    generate_fasta(prot_id, sequence, out_dir)
    command = '%s -query %s -db %s -out %s -evalue 0.001 -matrix BLOSUM62 -num_iterations 3 -num_threads 8' % (PSIBLAST, os.path.join(out_dir, prot_id + '.fasta'), UNIPROT_ALL_DB, os.path.join(out_dir, prot_id + '.rawmsa'))
    os.system(command)
    return os.path.join(out_dir, prot_id + '.rawmsa')

def parse_fasta(fasta_file):
    assert os.path.exists(fasta_file)
    cur_key = ''
    fasta_dict = {}
    keys_ordered = []
    for line in open(fasta_file, 'r'):
        if line[0] == '>':
            cur_key = line.strip().replace('>', '').split()[0]
            keys_ordered.append(cur_key)
            fasta_dict[cur_key] = ''
        else:
            fasta_dict[cur_key] += line.strip()	
    return keys_ordered, fasta_dict

def write_fasta(fasta_dict, fasta_file):
    output = open(fasta_file, 'w')
    for k, v in sorted(fasta_dict.items()):
        output.write('>%s\n%s\n' %(k, v))
    output.close()

def format_rawmsa(msa_files, out_dir):
    identifiers_to_aligns, outfiles, returned_file_dict = [], [], {}
    flag = False
    
    for e in msa_files:
        prot_id, msa_file = e[0], e[2]
        identifiers_to_align = set()
        with open(msa_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    tmp_id = line.strip().split()[0]
                    if tmp_id.split('|')[1] != prot_id:
                        identifiers_to_align.add(tmp_id[1:])
        identifiers_to_aligns.append(identifiers_to_align)
        if len(identifiers_to_align) > 0:
            outfiles.append(open(os.path.join(out_dir, prot_id+'_rawmsa.fasta'), 'w'))
            returned_file_dict[prot_id] = os.path.join(out_dir, prot_id+'_rawmsa.fasta')
            flag = True
        else:
            outfiles.append(None)
            returned_file_dict[prot_id] = None
        
    if flag:
        with open(UNIPROT_SEQS_PATH, 'r') as infile:
            for line in infile:
                line_list = line.strip().split()
                for i in range(len(identifiers_to_aligns)):
                    if line_list[0] in identifiers_to_aligns[i]:
                        outfiles[i].write('>' + line_list[0] + '\n' + line_list[1] +'\n')
                        
    for fh in outfiles:
        fh.close()
        
    return returned_file_dict

def run_cdhit(cdhit_input_file, cdhit_output_file):
    os.system('cd-hit -i '+cdhit_input_file+' -o '+cdhit_output_file+' -d 100')

def parse_cdhit_clstr(cdhit_input_file, cdhit_output_file):
    msa_dict, identifier2code = {}, {}
    with open(cdhit_input_file, 'r') as infile:
        lines = infile.readlines()
    if len(lines):
        for i in range(0, len(lines), 2):
            msa_dict[lines[i].strip().split('|')[1]] = lines[i+1].strip()
            identifier2code[lines[i].strip().split('|')[1]] = lines[i].strip().split('|')[2].split('_')[-1]
        
    clstr_info = {}
    with open(cdhit_output_file, 'r') as infile:
        for line in infile:
            if line[0] == '>':
                cluster_id = line.strip()[1:]
                clstr_info[cluster_id] = {}
            else:
                if line.strip()[-1] == '%':
                    clstr_info[cluster_id][line.strip().split('|')[1]] = float(line.strip().split()[-1][:-1])
                else:
                    clstr_info[cluster_id][line.strip().split('|')[1]] = 101
    return msa_dict, identifier2code, clstr_info

def run_clustal(clustal_input_file, clustal_output_file, num_threads=8):
    numseqs = 0
    try:
        with open(clustal_input_file, 'r') as f:
            for i, l in enumerate(f):
                numseqs = i
        f.close()
    except IOError:
        pass
    numseqs /= 2
    
    if numseqs > 1:
        # Run Clustal Omega
        os.system('%s -i %s -o %s --force --threads %s' % (CLUSTALO, clustal_input_file, clustal_output_file, str(num_threads)))
    else:
        # Clustal Omega will fail, copy single sequence to output file
        os.system('cp %s %s' % (clustal_input_file, clustal_output_file))
        
def format_clustal(clustal_output_file, formatted_output_file):
    msa_info = []
    with open(clustal_output_file, 'r') as f:
        seq_name = ''
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ''
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq.replace('U', '-'))
    # Make a temporary MSA file where the sequences are not split across multiple lines.
    # Generate the formatted CLUSTAL output.
    try:
        # Read clustal MSA
        outtxt = ''
        gaps = []
        # Iterate over each line
        for idx, line in enumerate(msa_info):
#             line = line.strip()
            # Add Header lines as they are
            if idx % 2 == 0:
                outtxt += line
                outtxt += '\n'
            # Special case for the first entry in the alignment
            # Find all of the gaps in the alignment since we only care about using the MSA with regard to the current UniProt
            # query. We don't care about any of the positions where the query has a gap
            elif idx == 1: # Query
                for i in range(len(line)): # Find all the Gaps
                    gaps.append(line[i] == '-')
            # For all matches
            if idx % 2 == 1:
                # Update the sequence by removing all of the positions that were a gap in the current UniProt alignment
                newseq = ''
                for i in range(len(gaps)):
                    if not gaps[i]:
                        if i < len(line):
                            newseq += line[i]
                        else:
                            newseq += '-'
                # Write the formatted alignment sequence
                outtxt += newseq
                outtxt += '\n'
        # Write all of the formatted alignment lines to the final alignment output
        with open(formatted_output_file, 'w') as f:
            f.write(outtxt)
    except IOError:
        pass

def cdhit_clstr_join(prot_id1, prot_id2, msa_fasta_file1, msa_fasta_file2, cdhit_output_clstr_file1, cdhit_output_clstr_file2, seq_dict, cdhit_clstr_join_output_path):
    msa_dict1, identifier2code1, clstr_info1 = parse_cdhit_clstr(msa_fasta_file1, cdhit_output_clstr_file1)
    if prot_id1 != prot_id2:
        msa_dict2, identifier2code2, clstr_info2 = parse_cdhit_clstr(msa_fasta_file2, cdhit_output_clstr_file2)
    else:
        msa_dict2, identifier2code2, clstr_info2 = msa_dict1, identifier2code1, clstr_info1
    
    all_pairs = set(identifier2code1.values()).intersection(set(identifier2code2.values()))
    if not all_pairs:
        return None
            
    joined_seqs = {}
    for k1 in sorted(list(clstr_info1.keys())):
        for k2 in sorted(list(clstr_info2.keys())):
            codes1 = [identifier2code1[ele] for ele in clstr_info1[k1]]
            codes2 = [identifier2code2[ele] for ele in clstr_info2[k2]]
            identical_codes = set(codes1).intersection(set(codes2))

            identifiers1 = sorted(clstr_info1[k1].items(), key=lambda x: x[1], reverse=True)
            identifiers2 = sorted(clstr_info2[k2].items(), key=lambda x: x[1], reverse=True)

            for identifier1 in identifiers1:
                for identifier2 in identifiers2:
                    if identifier2code1[identifier1[0]] == identifier2code2[identifier2[0]] and identifier2code1[identifier1[0]] in all_pairs:
                        all_pairs.remove(identifier2code1[identifier1[0]])
                        joined_seqs[identifier2code1[identifier1[0]]] = msa_dict1[identifier1[0]] + msa_dict2[identifier2[0]]

    if joined_seqs:
        cdhit_clstr_join_output_file = os.path.join(cdhit_clstr_join_output_path, prot_id1+'_'+prot_id2+'.cdhit_clstr_joined_seqs')
        with open(cdhit_clstr_join_output_file, 'w') as outfile:
            outfile.write('>'+prot_id1+'_'+prot_id2+'\n')
            outfile.write(seq_dict[prot_id1]+seq_dict[prot_id2]+'\n')

            for k in joined_seqs.keys():
                outfile.write('>'+k+'\n')
                outfile.write(joined_seqs[k]+'\n')
        return cdhit_clstr_join_output_file
    else:
        return None
