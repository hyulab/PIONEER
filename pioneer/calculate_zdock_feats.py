import os
import re
import glob
import pickle
import json
import argparse
from .utils import *
from .zdock import *
from .config import *
from tempfile import mkdtemp
from multiprocessing import Pool, Manager

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cpu', help='number of cores to use', type=int)
parser.add_argument('-f', '--feat', help='the feature to be calculated', type=str)
parser.add_argument('-i', '--input', help='input file containing docked models', type=str)
parser.add_argument('-o', '--output', help='path to output file', type=str)
parser.add_argument('-s', '--stype', help='type of structures to be docked', type=str)
parser.add_argument('-d', '--seq', help='dictionary of sequences', type=str)
args = parser.parse_args()

seq_dict = json.loads(args.seq)
if args.stype == 'pdb':
    docked_models = []
    with open(args.input, 'r') as f:
        for line in f:
            docked_models.append(line.strip())

    docked_model_files = []
    for e in docked_models:
        pdb_docked_files = glob.glob(e)
        docked_model_files += pdb_docked_files

    struct_ids = set()
    for f in docked_model_files:
        pdbA, subA, pdbB, subB, _, num, _ = re.split('--|-|\.', os.path.basename(f))
        struct_ids.add((pdbA[3:] + '_' + subA.split('_')[1]))
        struct_ids.add((pdbB[3:] + '_' + subB.split('_')[1]))

    sifts_data = parse_dictionary_list(PDBRESMAP_PATH)
    chain_to_sifts = {}
    for e in sifts_data:
        pdb_chain = e['PDB'] + '_' + e['Chain']
        if pdb_chain in struct_ids:
            if pdb_chain not in chain_to_sifts or len(unzip_res_range(e['MappableResInPDBChainOnUniprotBasis'])) > len(unzip_res_range(chain_to_sifts[pdb_chain]['MappableResInPDBChainOnUniprotBasis'])):
                chain_to_sifts[pdb_chain] = e

elif args.stype == 'mb':
    docked_models = []
    with open(args.input, 'r') as f:
        lines = f.read()
    line_list = lines.split('\nmodbase models\n')

    for line in line_list[0].strip().split('\n'):
        docked_models.append(line.strip())

    docked_model_files = []
    for e in docked_models:
        mb_docked_files = glob.glob(e)
        docked_model_files += mb_docked_files

    struct_ids = {}
    for f in docked_model_files:
        f_list = os.path.basename(f).split('--')

        if '_' in f_list[0]:
            struct_ids[f_list[0].split('_')[0]] = f_list[0]
        else:
            struct_ids[f_list[0]] = f_list[0]

        if '_' in f_list[1]:
            struct_ids[f_list[1].split('_')[0]] = f_list[1]
        else:
            struct_ids[f_list[1]] = f_list[1]

    chain_to_sifts = {}
    for p in line_list[1].strip().split('\n'):
        if os.path.isfile(p.strip()):
            for e in parse_dictionary_list(p.strip()):
                if e['modbase_modelID'].upper() in struct_ids:
                    chain_to_sifts[struct_ids[e['modbase_modelID'].upper()]] = e

else:
    docked_models = []
    with open(args.input, 'r') as f:
        lines = f.read()
    line_list = lines.split('\nmodbase models\n')

    for line in line_list[0].strip().split('\n'):
        docked_models.append(line.strip())

    docked_model_files = []
    for e in docked_models:
        mix_docked_files = glob.glob(e)
        docked_model_files += mix_docked_files

    struct_ids = set()
    struct_ids_mb = {}
    for f in docked_model_files:
        f_list = os.path.basename(f).split('--')

        if f_list[0].startswith('PDB') and f_list[0][7] == '.':
            tmp = f_list[0].split('_')
            pdbA = f_list[0][3:7]
            chainA = tmp[-1]
            struct_ids.add(pdbA+'_'+chainA)
        elif len(f_list[0].split('_')[0]) == 32:
            if '_' in f_list[0]:
                struct_ids_mb[f_list[0].split('_')[0]] = f_list[0]
            else:
                struct_ids_mb[f_list[0]] = f_list[0]

        if f_list[1].startswith('PDB') and f_list[1][7] == '.':
            tmp = f_list[1].split('_')
            pdbB = f_list[1][3:7]
            chainB = tmp[-1]
            struct_ids.add(pdbB+'_'+chainB)
        elif len(f_list[1].split('_')[0]) == 32:
            if '_' in f_list[1]:
                struct_ids_mb[f_list[1].split('_')[0]] = f_list[1]
            else:
                struct_ids_mb[f_list[1]] = f_list[1]

    sifts_data = parse_dictionary_list(PDBRESMAP_PATH)
    chain_to_sifts = {}
    for e in sifts_data:
        pdb_chain = e['PDB'] + '_' + e['Chain']
        if pdb_chain in struct_ids:
            if pdb_chain not in chain_to_sifts or len(unzip_res_range(e['MappableResInPDBChainOnUniprotBasis'])) > len(unzip_res_range(chain_to_sifts[pdb_chain]['MappableResInPDBChainOnUniprotBasis'])):
                chain_to_sifts[pdb_chain] = e

    for p in line_list[1].strip().split('\n'):
        if os.path.isfile(p.strip()):
            for e in parse_dictionary_list(p.strip()):
                if e['modbase_modelID'].upper() in struct_ids_mb:
                    uniprot_res = '[1-%s]' % (e['target_length'])
                    chain_to_sifts[struct_ids_mb[e['modbase_modelID'].upper()]] = {'UniProt': e['uniprot'], 'MappableResInPDBChainOnUniprotBasis': uniprot_res, 'MappableResInPDBChainOnPDBBasis': uniprot_res}

def calc_feat(f, q):
    if args.stype == 'pdb':
        pdbA, subA, pdbB, subB, _, num, _ = re.split('--|-|\.', os.path.basename(f))
        subA = pdbA[3:] + '_' + subA.split('_')[1]
        subB = pdbB[3:] + '_' + subB.split('_')[1]

        if subA not in chain_to_sifts or subB not in chain_to_sifts:
            return
        uniprotA = chain_to_sifts[subA]['UniProt']
        uniprotB = chain_to_sifts[subB]['UniProt']
        if uniprotA not in seq_dict or uniprotB not in seq_dict:
            return

        subA2uniprot = {}
        pdbres = unzip_res_range(chain_to_sifts[subA]['MappableResInPDBChainOnPDBBasis'])
        uniprotres = unzip_res_range(chain_to_sifts[subA]['MappableResInPDBChainOnUniprotBasis'])
        for i in range(len(pdbres)):
            subA2uniprot[pdbres[i]] = uniprotres[i]

        subB2uniprot = {}
        pdbres = unzip_res_range(chain_to_sifts[subB]['MappableResInPDBChainOnPDBBasis'])
        uniprotres = unzip_res_range(chain_to_sifts[subB]['MappableResInPDBChainOnUniprotBasis'])
        for i in range(len(pdbres)):
            subB2uniprot[pdbres[i]] = uniprotres[i]
        tmp_dir = mkdtemp()
        extract_single_model(f, os.path.join(tmp_dir, os.path.basename(f)))

        if args.feat == 'dist3d':
            out = chaindistcalc(os.path.join(tmp_dir, os.path.basename(f)), 'A', 'B')
            os.system('rm %s' % os.path.join(tmp_dir, os.path.basename(f)))
            if out is None:
                print('chaindistcalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                print('chaindistcalc failed: %s' %(f))
                return
            A_dist3d = dict([(int(subA2uniprot[q[1]]), q[3]) for q in out if q[1] in subA2uniprot and q[0]=='A'])
            B_dist3d = dict([(int(subB2uniprot[q[1]]), q[3]) for q in out if q[1] in subB2uniprot and q[0]=='B'])
            A_array = [str(A_dist3d[r]) if r in A_dist3d else 'nan' for r in range(1, len(seq_dict[uniprotA])+1)]
            B_array = [str(B_dist3d[r]) if r in B_dist3d else 'nan' for r in range(1, len(seq_dict[uniprotB])+1)]
        elif args.feat == 'ires':
            out = irescalc(os.path.join(tmp_dir, os.path.basename(f)), 'A', 'B', outfmt='residue_stats')
            os.system('rm %s' % os.path.join(tmp_dir, os.path.basename(f)))
            if out is None:
                print('irescalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                A_array, B_array = 'N/A', 'N/A'
            elif out[0][0].startswith('ERROR'):
                A_array, B_array = 'N/A', 'N/A'
            else:
                A_array = zip_res_range([subA2uniprot[q[1]] for q in out if q[1] in subA2uniprot and q[0] == 'A'])
                B_array = zip_res_range([subB2uniprot[q[1]] for q in out if q[1] in subB2uniprot and q[0] == 'B'])

    elif args.stype == 'mb':
        f_list = os.path.basename(f).split('--')
        subA = f_list[0]
        subB = f_list[1]
        num = f_list[2].split('-')[1].split('.')[0]
        if subA not in chain_to_sifts or subB not in chain_to_sifts:
            return
        uniprotA = chain_to_sifts[subA]['uniprot']
        uniprotB = chain_to_sifts[subB]['uniprot']

        if args.feat == 'dist3d':
            out = chaindistcalc(f, 'A', 'B')
            if out is None:
                print('chaindistcalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                print('chaindistcalc failed: %s' %(f))
                return
            A_dist3d = dict([(int(q[1]), q[3]) for q in out if q[0]=='A'])
            B_dist3d = dict([(int(q[1]), q[3]) for q in out if q[0]=='B'])
            A_array = [str(A_dist3d[r]) if r in A_dist3d else 'nan' for r in range(1, int(chain_to_sifts[subA]['target_length'])+1)]
            B_array = [str(B_dist3d[r]) if r in B_dist3d else 'nan' for r in range(1, int(chain_to_sifts[subB]['target_length'])+1)]
        elif args.feat == 'ires':
            out = irescalc(f, 'A', 'B', outfmt='residue_stats')
            if out is None:
                print('irescalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                A_array, B_array = 'N/A', 'N/A'
            elif out[0][0].startswith('ERROR'):
                A_array, B_array = 'N/A', 'N/A'
            else:
                A_array = zip_res_range([q[1] for q in out if q[0]=='A'])
                B_array = zip_res_range([q[1] for q in out if q[0]=='B'])

    else:
        f_list = os.path.basename(f).split('--')
        num = f_list[2].split('-')[1].split('.')[0]
        if f_list[0].startswith('PDB') and f_list[0][7] == '.':
            tmp = f_list[0].split('_')
            subA = f_list[0][3:7]+'_'+tmp[-1]
        elif len(f_list[0].split('_')[0]) == 32:
            subA = f_list[0]

        if f_list[1].startswith('PDB') and f_list[1][7] == '.':
            tmp = f_list[1].split('_')
            subB = f_list[1][3:7]+'_'+tmp[-1]
        elif len(f_list[1].split('_')[0]) == 32:
            subB = f_list[1]

        if subA not in chain_to_sifts or subB not in chain_to_sifts:
            return
        uniprotA = chain_to_sifts[subA]['UniProt']
        uniprotB = chain_to_sifts[subB]['UniProt']
        if uniprotA not in seq_dict or uniprotB not in seq_dict:
            return

        subA2uniprot = {}
        pdbres = unzip_res_range(chain_to_sifts[subA]['MappableResInPDBChainOnPDBBasis'])
        uniprotres = unzip_res_range(chain_to_sifts[subA]['MappableResInPDBChainOnUniprotBasis'])
        for i in range(len(pdbres)):
            subA2uniprot[pdbres[i]] = uniprotres[i]

        subB2uniprot = {}
        pdbres = unzip_res_range(chain_to_sifts[subB]['MappableResInPDBChainOnPDBBasis'])
        uniprotres = unzip_res_range(chain_to_sifts[subB]['MappableResInPDBChainOnUniprotBasis'])
        for i in range(len(pdbres)):
            subB2uniprot[pdbres[i]] = uniprotres[i]
        tmp_dir = mkdtemp()
        extract_single_model(f, os.path.join(tmp_dir, os.path.basename(f)))

        if args.feat == 'dist3d':
            out = chaindistcalc(os.path.join(tmp_dir, os.path.basename(f)), 'A', 'B')
            os.system('rm %s' % os.path.join(tmp_dir, os.path.basename(f)))
            if out is None:
                print('chaindistcalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                print('chaindistcalc failed: %s' %(f))
                return
            A_dist3d = dict([(int(subA2uniprot[q[1]]), q[3]) for q in out if q[1] in subA2uniprot and q[0]=='A'])
            B_dist3d = dict([(int(subB2uniprot[q[1]]), q[3]) for q in out if q[1] in subB2uniprot and q[0]=='B'])
            A_array = [str(A_dist3d[r]) if r in A_dist3d else 'nan' for r in range(1, len(seq_dict[uniprotA])+1)]
            B_array = [str(B_dist3d[r]) if r in B_dist3d else 'nan' for r in range(1, len(seq_dict[uniprotB])+1)]
        elif args.feat == 'ires':
            out = irescalc(os.path.join(tmp_dir, os.path.basename(f)), 'A', 'B', outfmt='residue_stats')
            os.system('rm %s' % os.path.join(tmp_dir, os.path.basename(f)))
            if out is None:
                print('irescalc failed: %s' %(f))
                return
            out = [x.strip().split('\t') for x in out.strip().split('\n')]
            if out == [['']]:
                A_array, B_array = 'N/A', 'N/A'
            elif out[0][0].startswith('ERROR'):
                A_array, B_array = 'N/A', 'N/A'
            else:
                A_array = zip_res_range([subA2uniprot[q[1]] for q in out if q[1] in subA2uniprot and q[0]=='A'])
                B_array = zip_res_range([subB2uniprot[q[1]] for q in out if q[1] in subB2uniprot and q[0]=='B'])
    try:
        zdock_score = open(f[:-7] + '.out').readlines()[int(num)+3].strip().split()[-1]
    except:
        zdock_score = 'n/a'
    # Final output as the return value
    if uniprotA > uniprotB:
        if args.feat == 'dist3d':
            q.put('\t'.join([uniprotB, uniprotA, num, subB, subA, os.path.basename(f), zdock_score, ';'.join(B_array), ';'.join(A_array)]) + '\n')
        elif args.feat == 'ires':
            q.put('\t'.join([uniprotB, uniprotA, num, subB, subA, os.path.basename(f), zdock_score, B_array, A_array]) + '\n')
    else:
        if args.feat == 'dist3d':
            q.put('\t'.join([uniprotA, uniprotB, num, subA, subB, os.path.basename(f), zdock_score, ';'.join(A_array), ';'.join(B_array)]) + '\n')
        elif args.feat == 'ires':
            q.put('\t'.join([uniprotB, uniprotA, num, subB, subA, os.path.basename(f), zdock_score, A_array, B_array]) + '\n')


def listener(q):
    output_f = open(args.output, 'w')
    output_f.write('\t'.join(['UniProtA', 'UniProtB', 'Rank', 'SubunitA', 'SubunitB', 'File', 'ZDOCK_Score', 'UniProtA_%s' % args.feat, 'UniProtB_%s' % args.feat]) + '\n')
    while 1:
        m = q.get()
        if m == 'kill':
            break
        output_f.write(m)
        output_f.flush()
    output_f.close()


manager = Manager()
q = manager.Queue()    
pool = Pool(args.cpu)

# Put listener to work first
watcher = pool.apply_async(listener, (q,))

# Fire off workers
jobs = []
for f in docked_model_files:
    job = pool.apply_async(calc_feat, (f, q))
    jobs.append(job)

# Collect results from the workers through the pool result queue
for job in jobs: 
    job.get()

# Now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()