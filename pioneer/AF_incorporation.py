import pandas as pd
import glob
import os
import gzip
import json
import pickle
from .config import AF_PRED_PATH

# New code to add AF predictions
def Add_AF(inter, UniProt, seq_dict, models_to_use, AF_predictions, Str = True):
    UniProt_AF = list(filter(lambda x: UniProt in x, AF_predictions))
    UniProt_AF.sort(key=lambda x: int(x.split('-')[2][1:]))
    if len(UniProt_AF) == 0:
        return models_to_use
    with gzip.open(UniProt_AF[-1], 'rb') as f:
        line = f.readlines()[-3].decode('utf-8')
    if 200*(len(UniProt_AF)-1)+int(line.strip()[-4:]) != len(seq_dict[UniProt]):
        return models_to_use
    if Str == True:
        if UniProt_AF:
            for AF in UniProt_AF:
                models_to_use[inter][UniProt].append(['AF', AF])
    else: # for protein without structure information where its partner has structure information or both proteins do not have structure information
        Str_list = []
        inter_with_str = set(models_to_use.keys())
        if inter in inter_with_str: # for protein without structure information where its partner has structure information
            if UniProt_AF:
                for AF in UniProt_AF:
                    Str_list.append(['AF', AF])
                models_to_use[inter][UniProt] = Str_list
        else: # for protein where none of proteins in interaction has structure information
            if UniProt_AF:
                models_to_use[inter] = {}
                for AF in UniProt_AF:
                    Str_list.append(['AF', AF])
                models_to_use[inter][UniProt] = Str_list
    return models_to_use

def AF_incort(int_list, seq_dict, models_to_use):
    # All AF predictions
    """
    AF_predictions = []
    AF_prediction_path = '/fs/cbsuhyfs1/storage/resources/alphafold/data/'
    species = os.listdir(AF_prediction_path)
    for s in species:
        predictions = glob.glob(AF_prediction_path + s + '/*.pdb.gz')
        AF_predictions += predictions
    """
    print("Loading AF paths")
    with open(AF_PRED_PATH, 'rb') as f:
        AF_predictions = pickle.load(f)

    inter_with_str = models_to_use.keys()
    for i, inter in enumerate(int_list):
        p1, p2 = inter[0], inter[1]
        if inter in inter_with_str: # at least one protein already has structure information
            if len(models_to_use[inter]) == 2: # Heterodimer, both proteins have structure information
                models_to_use = Add_AF(inter, p1, seq_dict, models_to_use, AF_predictions, Str = True)
                models_to_use = Add_AF(inter, p2, seq_dict, models_to_use, AF_predictions, Str = True)
            else: # Only one protein has structure information or homodimer
                if p1 == p2: # Homodimer
                    models_to_use = Add_AF(inter, p1, seq_dict, models_to_use, AF_predictions, Str = True)
                else:
                    key_prot = list(models_to_use[inter].keys())
                    if p1 == key_prot[0]: # if only p1 has structure information
                        models_to_use = Add_AF(inter, p1, seq_dict, models_to_use, AF_predictions, Str = True)
                        models_to_use = Add_AF(inter, p2, seq_dict, models_to_use, AF_predictions, Str = False)
                    else:
                        models_to_use = Add_AF(inter, p1, seq_dict, models_to_use, AF_predictions, Str = False)
                        models_to_use = Add_AF(inter, p2, seq_dict, models_to_use, AF_predictions, Str = True)
        else: # both proteins do not have structure information
            models_to_use = Add_AF(inter, p1, seq_dict, models_to_use, AF_predictions, Str = False)
            models_to_use = Add_AF(inter, p2, seq_dict, models_to_use, AF_predictions, Str = False)
    return models_to_use
