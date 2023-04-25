import sys
import warnings

import numpy as np
import scipy as sp
from numba import jit
from numba.core.errors import NumbaWarning
from scipy.sparse import csr_matrix as sparsify
from scipy.spatial.distance import squareform, pdist
warnings.filterwarnings('ignore', category=NumbaWarning)

##################################################################################
# Helper functions from pySCA
# Source: https://github.com/reynoldsk/pySCA/blob/master/scaTools.py

def readAlg(filename):
    ''' Read in a multiple sequence alignment in fasta format, and return the 
    headers and sequences.
    >>> headers, sequences = readAlg(filename) '''
    filelines = open(filename, 'r').readlines()
    headers = list(); sequences = list(); notfirst = 0
    for line in filelines:
        if line[0] == '>':
            if notfirst > 0: sequences.append(seq.replace('\n','').upper())
            headers.append(line[1:].replace('\n',''))
            seq = ''; notfirst = 1
        elif line != '\n': seq += line
    sequences.append(seq.replace('\n','').upper())
    return headers, sequences


def lett2num(msa_lett, code='ACDEFGHIKLMNPQRSTVWY'):
    ''' Translate an alignment from a representation where the 20 natural amino
    acids are represented by letters to a representation where they are
    represented by the numbers 1,...,20, with any symbol not corresponding to an
    amino acid represented by 0.
    :Example:
       >>> msa_num = lett2num(msa_lett, code='ACDEFGHIKLMNPQRSTVWY') 
    '''
    lett2index = {aa:i+1 for i,aa in enumerate(code)}
    [Nseq, Npos] = [len(msa_lett), len(msa_lett[0])]
    msa_num = np.zeros((Nseq, Npos)).astype(int)
    for s, seq in enumerate(msa_lett):
        for i, lett in enumerate(seq):
            if lett in code:
                 msa_num[s, i] = lett2index[lett]
    return msa_num


def alg2bin(alg, N_aa=20):
    ''' Translate an alignment of size M x L where the amino acids are represented 
    by numbers between 0 and N_aa (obtained using lett2num) to a sparse binary 
    array of size M x (N_aa x L). 
    
    :Example:
      >>> Abin = alg2bin(alg, N_aa=20) '''
    
    [N_seq, N_pos] = alg.shape
    Abin_tens = np.zeros((N_aa, N_pos, N_seq))
    for ia in range(N_aa):
        Abin_tens[ia,:,:] = (alg == ia+1).T
    Abin = sparsify(Abin_tens.reshape(N_aa*N_pos, N_seq, order='F').T)
    return Abin

##################################################################################
# Helper functions for SCA calculation
# Modified based on the pySCA Python package

@jit
def freq(al2d, freq0=np.array([.073,.025,.050,.061,.042,.072,.023,.053,.064,.089,.023,.043,.052,.040,.052,.073,.056,.063,.013,.033])):
    Nseq = al2d.shape[0]
    Npos = al2d.shape[1] // 20
    seqwn = np.ones((1, Nseq)) / Nseq
    freq1 = np.dot(seqwn, np.array(al2d.todense()))[0]
    return freq1


@jit
def posWeights(al2d, freq0=np.array([.073,.025,.050,.061,.042,.072,.023,.053,.064,.089,.023,.043,.052,.040,.052,.073,.056,.063,.013,.033]), 
                   tolerance=1e-12):
    Nseq = al2d.shape[0]
    Npos = al2d.shape[1] // 20
    seqw = np.ones((1, Nseq))
    freq1 = freq(al2d)
    
    # Overall fraction of gaps:
    theta = 1 - np.sum(freq1) / Npos
    if theta < tolerance:
        theta = 0
    freqg0 = (1 - theta) * freq0
    freq0v = np.tile(freq0, Npos)
    iok = [i for i in range(Npos * 20) if (freq1[i] > 0 and freq1[i] < 1)]
    Wia = np.zeros(Npos * 20)
    Wia[iok] = abs(np.log((freq1[iok] * (1 - freq0v[iok])) / ((1 - freq1[iok]) * freq0v[iok])))
    return Wia

##################################################################################
# Helper functions for DCA calculation
# Written based on /home/mjm659/mjm_path/dca.m on multivac

@jit
def compute_true_frequencies(align, M, N, q, theta):
    W = 1.0 / (1 + np.sum(squareform(pdist(align, 'hamming') < theta), axis=0))
    Meff = np.sum(W)
    
    Pi_true = np.zeros((N,q))
    for i in range(N):
        for j in range(M):
            Pi_true[i, align[j,i]-1] += W[j]
    Pi_true /= Meff
    
    Pij_true = np.zeros((N,N,q,q))
    for l in range(M):
        for i in range(N-1):
            for j in range(i, N): 
                Pij_true[i,j,align[l,i]-1,align[l,j]-1] += W[l]
                Pij_true[j,i,align[l,j]-1,align[l,i]-1] = Pij_true[i,j,align[l,i]-1,align[l,j]-1] 
    Pij_true /= Meff
    for i in range(N):
        Pij_true[i,i,:,:] = np.diag(Pi_true[i])
    return Pij_true, Pi_true, Meff


@jit
def with_pc(Pij_true, Pi_true, pseudocount_weight, N, q):
    Pij = (1 - pseudocount_weight) * Pij_true + pseudocount_weight / q / q
    Pi = (1 - pseudocount_weight) * Pi_true + pseudocount_weight / q
    for i in range(N):
         Pij[i,i,:,:] = (1 - pseudocount_weight) * Pij_true[i,i,:,:] + np.identity(q) * pseudocount_weight / q
    return Pij, Pi


@jit
def compute_C(Pij, Pi, N, q):
    C = np.zeros((N*(q-1), N*(q-1)))
    for i in range(N):
        for j in range(N):
            for a in range(q-1):
                for b in range(q-1):
                    C[(q-1)*i+a, (q-1)*j+b] = Pij[i,j,a,b] - Pi[i,a] * Pi[j,b]
    return C


@jit
def calculate_mi(i, j, Pij, Pi, q):
    M = 0
    for a in range(q):
        for b in range(q):
            if Pij[i,j,a,b] > 0:
                M += Pij[i,j,a,b] * np.log(Pij[i,j,a,b] / Pi[i,a] / Pi[j,b])
    s1 = 0
    s2 = 0
    for a in range(q):
        if Pi[i,a] > 0:
            s1 -= Pi[i,a] * np.log(Pi[i,a])
        if Pi[j,a] > 0:
            s2 -= Pi[j,a] * np.log(Pi[j,a])
    return M, s1, s2


@jit
def numba_inv(M):
    return np.linalg.inv(M)


@jit
def returnW(C, i, j, q):
    W = np.ones((q,q))
    rows = np.arange((q-1)*i, (q-1)*(i+1))
    cols = np.arange((q-1)*j, (q-1)*(j+1))
    sub = C[np.ix_(rows, cols)]
    W[:q-1,:q-1] = np.exp(-sub)
    return W


@jit
def compute_mu(i, j, W, Pi, q):
    diff = 1.0
    mu1 = np.ones(q) / q
    mu2 = np.ones(q) / q
    pi = Pi[i]
    pj = Pi[j]    
    while(diff > 1e-4):
        scra1 = np.dot(W, mu2)
        scra2 = np.dot(W.T, mu1)
        new1 = pi / scra1
        new1 /= new1.sum()
        new2 = pj / scra2
        new2 /= new2.sum()
        diff = np.max(np.maximum(np.abs(new1 - mu1), np.abs(new2 - mu2)))
        mu1 = new1
        mu2 = new2
    return mu1, mu2


@jit
def compute_di(i, j, W, mu1, mu2, Pi):
    Pdir = W * np.matmul(mu1[np.newaxis,:].T, mu2[np.newaxis,:])
    Pdir /= np.sum(Pdir)
    Pfac = np.matmul(Pi[i][np.newaxis,:].T, Pi[j][np.newaxis,:])
    DI = np.einsum('ij,ji->', Pdir.T, np.log(np.divide(Pdir+1e-100, Pfac+1e-100)))
    return DI


def compute_results(Pij, Pi, Pij_true, Pi_true, invC, N, q, file_handle):
    for i in range(N-1):
        for j in range(i+1, N):
            MI_true, si_true, sj_true = calculate_mi(i, j, Pij_true, Pi_true, q)
            if MI_true < 1.01 * np.finfo(float).eps:
                MI_true = 0
            W_mf = returnW(invC, i, j, q)
            mu1, mu2 = compute_mu(i, j, W_mf, Pi, q)
            DI_mf_pc = compute_di(i, j, W_mf, mu1, mu2, Pi)
            if DI_mf_pc < 1.01 * np.finfo(float).eps:
                DI_mf_pc = 0
            file_handle.write('%d %d %g %g\n' % (i+1, j+1, MI_true, DI_mf_pc))


##################################################################################
# Functions to calculate SCA and DCA

def sca(joined_msa_file, out_file):
    aln = lett2num(readAlg(joined_msa_file)[1])
    N_seq, N_pos = aln.shape
    
    al2d = alg2bin(aln)
    W = posWeights(al2d)
    
    wX = np.asarray(np.multiply(al2d.todense(), W)).reshape((N_seq, N_pos, 20))
    W = W.reshape((N_pos, 20))
    abin = np.zeros((N_seq, N_pos, 20))
    
    for i in range(1,21):
        abin[:,:,i-1] = (aln == i)
    freq = (np.sum(abin, axis=0) / N_seq).T
    pm = np.multiply(freq.T, W)
    
    for i in range(N_pos):
        if not np.all(pm[i] == 0):
            pm[i] = pm[i] / np.linalg.norm(pm[i])
    pwX = np.einsum('ij,kij->ki', pm, wX)
    Cp = np.abs(np.matmul(pwX.T, pwX) / aln.shape[0] - np.matmul(np.mean(pwX, axis=0, keepdims=True).T, np.mean(pwX, axis=0, keepdims=True)))
    
    with open(out_file, 'w') as f:
        for i in range(N_pos-1):
            for j in range(i+1, N_pos):
                f.write('%i\t%i\t%f\n' % (i+1, j+1, Cp[i,j]))


def dca(joined_msa_file, out_file):
    pseudocount_weight = 0.5    # relative weight of pseudo count   
    theta = 0.2                 # threshold for sequence id in reweighting
    
    align = lett2num(readAlg(joined_msa_file)[1]) + 1
    M, N = align.shape
    q = np.max(align)
    
    Pij_true, Pi_true, Meff = compute_true_frequencies(align, M, N, q, theta)
    Pij, Pi = with_pc(Pij_true, Pi_true, pseudocount_weight, N, q)
    C = compute_C(Pij, Pi, N, q)
    invC = numba_inv(C)
    
    with open(out_file, 'w') as f:
        compute_results(Pij, Pi, Pij_true, Pi_true, invC, N, q, f)