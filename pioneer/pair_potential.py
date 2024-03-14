import os
import glob
import numpy as np

class PairPotential:
    """
    Pair potential class written by Dapeng Xiong.
    """
    
    def __init__(self):
        self.AA2I = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 
                     'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.I2AA = {0:'A', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L',
                     10:'M', 11:'N', 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y'}
        self.AA2V = {'A':88.6, 'C':108.5, 'D':111.1, 'E':138.4, 'F':189.9, 'G':60.1, 'H':153.2, 'I':166.7, 'K':168.6, 'L':166.7,
                     'M':162.9, 'N':114.1, 'P':112.7, 'Q':143.8, 'R':173.4, 'S':89.0, 'T':116.1, 'V':140.0, 'W':227.8, 'Y':193.6}
    
    
    def get_potential(self, dimers, uniprot2seq, dimer2ires):
        """
        Calculate background amino acid pair distribution.
        """
        dimer2ires_indices = {}
        for dimer in dimers:
            dimer2ires_indices[dimer] = []
            for ires in dimer2ires[dimer]:
                ind = []
                for i in range(len(ires)):
                    if ires[i] == 1:
                        ind.append(i)
                dimer2ires_indices[dimer].append(ind)
        W = np.zeros((20,))
        for dimer in dimers:
            for p in [0, 1]:
                for i in dimer2ires_indices[dimer][p]:
                    if uniprot2seq[dimer.split('_')[p]][i] in self.AA2I:
                        W[self.AA2I[uniprot2seq[dimer.split('_')[p]][i]]] += 1
                    else:
                        continue
        W = W / np.sum(W)
        Q = np.zeros((20, 20))
        for dimer in dimers:
            p1, p2 = dimer.split('_')
            for i in dimer2ires_indices[dimer][0]:
                if uniprot2seq[p1][i] in self.AA2I:
                    N_i = self.AA2I[uniprot2seq[p1][i]]
                else:
                    continue
                for j in dimer2ires_indices[dimer][1]:
                    if uniprot2seq[p2][j] in self.AA2I:
                        N_j = self.AA2I[uniprot2seq[p2][j]]
                    else:
                        continue
                    if N_i <= N_j:
                        Q[N_i][N_j] += 1
                    else:
                        Q[N_j][N_i] += 1
        for N_i in range(20):
            for N_j in range(N_i, 20):
                Q[N_i][N_j] = Q[N_i][N_j] * self.AA2V[self.I2AA[N_i]] * self.AA2V[self.I2AA[N_j]]
        Q = Q/np.sum(Q)
        G = np.zeros((20, 20))
        for N_i in range(20):
            for N_j in range(N_i, 20):
                G[N_i][N_j] = np.log(Q[N_i][N_j]/(W[N_i]*W[N_j]))
                if N_i != N_j:
                    G[N_j][N_i] = G[N_i][N_j]
        return G
    
    
    def get_feature(self, prot_seq1, prot_seq2, G):
        """
        Calculate pair potential feature values.
        
        Args:
            prot_seq1 (str): Sequence of the first protein.
            prot_seq2 (str): Sequence of the second protein.
            G (ndarray): Background amino acid pair distribution.
            
        Returns:
            An array representing pair potential feature values.
        
        """
        AA_freq = np.zeros((20,))
        for i in self.I2AA:
            AA_freq[i] = 1.0 * prot_seq2.count(self.I2AA[i]) / len(prot_seq2)
        pair_potential = np.zeros((20,))
        for i in range(20):
            pair_potential[i] = np.sum(G[i, :] * AA_freq)
        pair_potential_feature = []
        for AA in prot_seq1:
            if AA in self.AA2I:
                pair_potential_feature.append(pair_potential[self.AA2I[AA]])
            else:
                pair_potential_feature.append(np.nan)
        return np.array(pair_potential_feature)


def normalize_feat(feat_array):
    """
    Calculate normalized pair potential feature.
    
    Args:
        feat_array (ndarray): Raw pair potential feature array.
    
    Returns:
        An array representing normalized pair potential feature values.
    
    """
    if abs(np.nanstd(feat_array) - 0.0) > 1e-9:
        normed_array = (feat_array - np.nanmean(feat_array)) / np.nanstd(feat_array)
    else:
        normed_array = (feat_array - np.nanmean(feat_array)) / (np.nanstd(feat_array) + 1e-9)
    return normed_array