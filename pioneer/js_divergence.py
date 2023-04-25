import os
import numpy as np
from copy import deepcopy

iupac_alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','U','V','W','Y','Z','X','*','-']
aa_to_index = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9,
               'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, '-':20}


def read_fasta_alignment(msa_file):
    """
    Read in the alignment stored in fasta format. Return an ndarray of aligned sequences.
    
    Args:
      msa_file: str, path to the alignment fasta file.
    """
    alignment = []
    with open(msa_file, 'r') as f:
        msa_content = f.read()
    for to_process in msa_content.split('>')[1:]:
        to_process_list = to_process.split('\n')
        sequence = ''.join(to_process_list[1:]).upper().replace('B', 'D').replace('Z', 'Q').replace('X', '-')
        sequence = ''.join([c if c in iupac_alphabet else '-' for c in sequence])
        alignment.append(list(sequence.replace('U', '-')))

    alignment = np.asarray(alignment, dtype=str)
    
    # Sanity check: all sequences must have the same length.
    # Remove all identifiers and sequences that do not have the same length as the human sequence.
    to_remove = []
    if alignment.shape[0] > 0:
        human_seq = alignment[0]
    for i, seq in enumerate(alignment):
        if len(seq) != len(human_seq):
            to_remove.append(i)
    if len(to_remove) > 0:
        to_remove = np.asarray(to_remove)
        mask = np.ones(alignment.shape[0], dtype=bool)
        mask[to_remove] = False
        alignment = alignment[mask]
    return alignment


def calculate_sequence_weights(alignment):
    """
    Calculate the sequence weights using the Henikoff '94 method for the given MSA.
    
    Args:
      alignment: np.array, numpy array representing the alignment.
    """
    seq_weights = np.zeros(alignment.shape[0], dtype=float)
    for i in range(alignment.shape[1]):          # all positions
        freq_counts = np.zeros(21, dtype=float)
        for j in range(alignment.shape[0]):      # all sequences
            if alignment[j][i] != '-':           # ignore gaps
                freq_counts[aa_to_index[alignment[j,i]]] += 1
        num_observed_types = np.nonzero(freq_counts)[0].shape[0]
        for j in range(alignment.shape[0]):      # all sequences
            d = freq_counts[aa_to_index[alignment[j,i]]] * num_observed_types
            if d > 0:
                seq_weights[j] += 1/d
    seq_weights = seq_weights / alignment.shape[0]
    return seq_weights


def weighted_freq_count_pseudocount(col, seq_weights, pc_amount):
    """
    Return the weighted frequency count for a column with pseudocount.
    
    Args:
      col:         np.array, a column in the MSA.
      seq_weights: np.array, weights for each sequence in the alignment.
      pc_amount:   float, pseudocount.
    """
    # If the weights do not match, use equal weight.
    if len(seq_weights) != len(col):
        seq_weights = np.ones(len(col), dtype=float)
    
    freq_counts = np.array([pc_amount] * 21) # For each AA
    for j in range(len(col)):
        freq_counts[aa_to_index[col[j]]] += 1 * seq_weights[j]
    freq_counts = freq_counts / (np.sum(seq_weights) + 21 * pc_amount)
    return freq_counts


def weighted_gap_penalty(col, seq_weights):
    """
    Calculate the simple gap penalty multiplier for the column. If the sequences are weighted, 
    the gaps, when penalized, are weighted accordingly.
    
    Args:
      col:         np.array, a column in the MSA.
      seq_weights: np.array, weights for each sequence in the alignment.
    """
    # If the weights do not match, use equal weight.
    if len(seq_weights) != len(col):
        seq_weights = np.ones(len(col), dtype=float)
    
    gap_sum = np.sum(seq_weights[np.where(np.array(col) == '-')[0]])
    return 1 - gap_sum / np.sum(seq_weights)


def js_divergence(col, bg_distr, seq_weights, pc_amount, gap_penalty=1):
    """
    Calculate JS divergence for one column in the MSA.
    
    Args:
      col: np.array, a column in the MSA.
      bg_distr: np.array, background distribution of the 20 amino acids.
      seq_weights: np.array, weights for each sequence in the alignment.
      pc_amount: float, pseudocount.
      gap_penalty: float, gap penalty.
    """
    fc = weighted_freq_count_pseudocount(col, seq_weights, pc_amount)
    
    # If background distribution lacks a gap count, remove fc gap count.
    if len(bg_distr) == 20:
        fc = fc[:-1]
        fc = fc / np.sum(fc)
    
    # Make r distribution
    r = 0.5 * fc + 0.5 * bg_distr
    d = 0
    for i in range(r.shape[0]):
        if r[i] != 0:
            if fc[i] == 0:
                d += bg_distr[i] * np.log2(bg_distr[i]/r[i])
            elif bg_distr[i] == 0:
                d += fc[i] * np.log2(fc[i]/r[i]) 
            else:
                d += fc[i] * np.log2(fc[i]/r[i]) + bg_distr[i] * np.log2(bg_distr[i]/r[i])
    d /= 2
    if gap_penalty == 1: 
        return d * weighted_gap_penalty(col, seq_weights)
    else: 
        return d
    
    
def window_score(scores, window_len, lam=0.5):
    """
    This function takes a list of scores and a length and transforms them so that each position 
    is a weighted average of the surrounding positions. Positions with scores less than zero are not 
    changed and are ignored in the calculation. Here window_len is interpreted to mean window_len 
    residues on either side of the current residue.
    
    Args:
      scores: list, list of scores.
      window_len: int, length of the window.
      lam: float, weight given to the original scores.
    """
    w_scores = deepcopy(scores)
    for i in range(window_len, len(scores) - window_len):
        if scores[i] < 0:
            continue
        score_sum = 0
        num_terms = 0
        for j in range(i - window_len, i + window_len + 1):
            if i != j and scores[j] >= 0:
                num_terms += 1
                score_sum += scores[j]
        if num_terms > 0:
            w_scores[i] = (1 - lam) * (score_sum / num_terms) + lam * scores[i]
    return w_scores


def calculate_js_div_from_msa(msa_file, bg_distr, pc_amount, window_size, lam):
    """
    Calculate JS divergence from a multiple sequence alignment.
    
    Args:
      msa_file: str, path to the alignment fasta file.
      bg_distr: np.array, background distribution of the 20 amino acids.
      pc_amount: float, pseudocount.
      window_size: int, length of the window.
      lam: float, weight given to the original scores.
    """
    alignment = read_fasta_alignment(msa_file)
    seq_weights = calculate_sequence_weights(alignment)
    
    score_list = []
    for i in range(alignment[0].shape[0]):
        col = alignment[:,i]
        score_list.append(js_divergence(col, bg_distr, seq_weights, pc_amount, 1))
    
    if window_size > 0:
        scores = window_score(score_list, window_size, lam)
    return np.array(scores)