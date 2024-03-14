import os, sys, glob, json
import pandas as pd
from pioneer.compile_features import *
from pioneer.prediction import *
from pioneer.data_generation import *
from pioneer.integration import *
from pioneer.AF_incorporation import *
from pioneer.models_to_use_cleaning import *
from datetime import datetime, timedelta
from random import choice
from string import ascii_lowercase, digits
import urllib

if len(sys.argv) != 3:
    print('Usage: python {0} <interactions_to_process> <result_folder_name>'.format(sys.argv[0]))
    exit(1)

#input
int_list = []
with open(sys.argv[1]) as f:
    for line in f:
        line_list = line.strip().split()
        if len(line_list) != 2:
            print('Each line should contain a pair of uniprot IDs.')
            exit(1)
            
        p1, p2 = line_list
        if p1 <= p2:
            int_list.append((p1, p2))
        else:
            int_list.append((p2, p1))
    
uniprots = set()
for interaction in int_list:
    p1, p2 = interaction
    uniprots.add(p1)
    uniprots.add(p2)
    
seq_dict = {}
for uniprot in uniprots:
    try:
        response = urllib.request.urlopen('https://www.uniprot.org/uniprot/'+uniprot+'.fasta')
        response = response.read().decode('utf-8').strip().split('\n')
        seq = ''.join(response[1:])
        if seq:
            seq_dict[uniprot] = seq
    except:
        print('%s does not exist in uniprot database' % uniprot)
    
to_remove = set()
for interaction in int_list:
    p1, p2 = interaction
    if p1 not in seq_dict or p2 not in seq_dict:
        to_remove.add(interaction)
for interaction in to_remove:
    int_list.remove(interaction)
    
uniprots = set()
for interaction in int_list:
    p1, p2 = interaction
    uniprots.add(p1)
    uniprots.add(p2)
to_remove = set(seq_dict.keys())-uniprots
for uniprot in to_remove:
    seq_dict.pop(uniprot)
# end input

#make directory for user's results
result_folder_name = sys.argv[2]
folder_name = os.getcwd()+'/'+result_folder_name
if not os.path.exists(folder_name):
    os.system('mkdir %s' % folder_name)
else:
    print('%s already exists.' % result_folder_name)
    exit(1)
print('The results and relevant files can be found in %s' % folder_name)

# parameters
AF_threshold = 80
# end parameters

# paths
non_cached_dir = folder_name+'/non_cached_files/' # Directory containing each feature and its associated files
struct_data_dir = folder_name+'/structs/' # Directory containing structural information
seq_feat_dir = folder_name+'/seq_feats/' # Directory containing sequence feature files calculated from `compile_pioneer_seq_feats`
full_feat_dir = folder_name+'/full_feats/' # Directory containing full feature files calculated from `compile_pioneer_full_feats`
seq_seq_prediction_dir = folder_name+'/seq_seq_prediction' # Directory for prediction files made by seq_seq model
seq_str_prediction_dir = folder_name+'/seq_str_prediction' # Directory for prediction files made by seq_str model
str_seq_prediction_dir = folder_name+'/str_seq_prediction' # Directory for prediction files made by str_seq model
str_str_prediction_dir = folder_name+'/str_str_prediction' # Directory for prediction files made by str_str model
PIONEER_prediction_dir = folder_name+'/PIONEER_prediction' # Directory for final prediction
# paths

# Initialize (empty) all directories
dirs = [non_cached_dir, struct_data_dir, seq_feat_dir, full_feat_dir, seq_seq_prediction_dir, seq_str_prediction_dir, str_seq_prediction_dir, str_str_prediction_dir, PIONEER_prediction_dir]
for direc in dirs:
    os.system('mkdir -p %s' % direc)
print('Directories initialized')

compile_pioneer_seq_feats(int_list, seq_dict, non_cached_dir, seq_feat_dir)
models_to_use = compile_pioneer_full_feats(int_list, seq_dict, non_cached_dir, seq_feat_dir, full_feat_dir, exclude_hom_struct=True)
print('Feature generated')

seq_seq, seq_str, str_seq, str_str = obtain_all_structures(int_list, models_to_use, non_cached_dir, struct_data_dir)
print('Structures collected')

print('Deep Learning input generates')
seq_seq_data = generate_seq_seq_data(seq_seq, seq_feat_dir) # seq seq prediction
# seq str prediction for interactions that only p2 has structure
seq_str_data_p1 = generate_seq_str_data(seq_str, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=False, AF_threshold=AF_threshold)
# seq str prediction for interactions that p1 and p2 both have structure. We need this for residues in p1 that are not predicted by str str model (interactions p1 and p2 both have structure)
seq_str_data_p1_backup = generate_seq_str_data(str_str, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=False, AF_threshold=AF_threshold)
# seq str prediction for interactions that only p1 has structure (We predict for residues in p2). So, a list of str_seq is an input
seq_str_data_p2 = generate_seq_str_data(str_seq, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=True, AF_threshold=AF_threshold)
# seq str prediction for interactions that p1 and p2 both have structure. We need this for residues in p2 that are not predicted by str str model (interactions p1 and p2 both have structure)
seq_str_data_p2_backup = generate_seq_str_data(str_str, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=True, AF_threshold=AF_threshold)
# str seq prediction for interactions that only p1 has structure
str_seq_data_p1 = generate_str_seq_data(str_seq, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=False, AF_threshold=AF_threshold)
# str seq prediction for interactions that only p2 has structure
str_seq_data_p2 = generate_str_seq_data(seq_str, seq_feat_dir, full_feat_dir, struct_data_dir, non_cached_dir, p2=True, AF_threshold=AF_threshold)
str_str_data = generate_str_str_data(str_str, full_feat_dir, struct_data_dir, non_cached_dir, AF_threshold=AF_threshold)

device = 'cpu' #If you want to use GPU 0, change this to 'cuda:0'
print('Prediction starts')
seq_seq_prediction(seq_seq_data, seq_seq_prediction_dir, device)
seq_str_prediction(seq_str_data_p1, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p1_backup, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p2, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p2_backup, seq_str_prediction_dir, device)
str_seq_prediction(str_seq_data_p1, str_seq_prediction_dir, device)
str_seq_prediction(str_seq_data_p2, str_seq_prediction_dir, device)
str_str_prediction(str_str_data, str_str_prediction_dir, device)

integration(seq_seq_prediction_dir, seq_str_prediction_dir, str_seq_prediction_dir, str_str_prediction_dir, PIONEER_prediction_dir)

print('End PIONEER')
