import itertools
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_residue_info(pdb_array):
    """
    Parse out row number boundaries for each residue from input array.
    
    Args:
        pdb_array (array): 2D array of atom information from PDB file.
    
    Returns:
        A 2-D array of width 2 where each row has the row number boundaries for each residue.
    
    """
    atom_res_array = pdb_array[:,6]
    boundary_list = []
    start_pointer = 0
    curr_pointer = 0
    curr_atom = atom_res_array[0]
    
    # One pass through the list of residue numbers and record row number boundaries. Both sides inclusive.
    while(curr_pointer < atom_res_array.shape[0] - 1):
        curr_pointer += 1
        if atom_res_array[curr_pointer] != curr_atom:
            boundary_list.append([start_pointer, curr_pointer - 1])
            start_pointer = curr_pointer
            curr_atom = atom_res_array[curr_pointer]
    boundary_list.append([start_pointer, atom_res_array.shape[0] - 1])
    return np.array(boundary_list)


def get_distance_matrix(pdb_array, residue_index, distance_type):
    """
    Calculate distance matrix for all residues.
    
    Args:
        pdb_array (array): 2D array of atom information from PDB file.
        residue_index (array): 2D array specifying row number boundaries for each residue.
        distance_type (str): Method for calculating residue distance.
      
    Returns:
        A 2D array of distances between residues.
    
    """
    if distance_type == 'atoms_average':
        full_atom_dist = squareform(pdist(pdb_array[:,7:10].astype(float)))
        residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
        for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
            index_i = residue_index[i]
            index_j = residue_index[j]
            distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1]+1,index_j[0]:index_j[1]+1])
            residue_dm[i][j] = distance_ij
            residue_dm[j][i] = distance_ij
    elif distance_type == 'centroid':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            res_start, res_end = residue_index[i]
            coord_i = pdb_array[:,7:10][res_start:res_end+1].astype(np.float)
            coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
    
    else:
        raise ValueError('Invalid distance type: %s' % distance_type)
    return residue_dm


def get_neighbor_index(residue_dm, num_neighbors):
    """
    Obtain indices of `num_neighbors` nearest neighbors.
    
    Args:
        residue_dm (array): 2D array of distances between residues.
        num_neighbors (int): Number of nearest neighbors.
      
    Returns:
        A 2D array of shape [num_residues, num_neighbors].
    
    """
    return residue_dm.argsort()[:,1:num_neighbors+1]


def get_edge_coo_data(pdb_array, num_neighbors, distance_type='atoms_average'):
    """
    Calculates graph-related data required for the graph neural network in PyTorch Geometric.
    
    Args:
        pdb_array (array): 2D array of atom information from PDB file.
        num_neighbors (int): Number of nearest neighbors.
        distance_type (str): Method for calculating residue distance.
    
    Returns:
        An np.ndarray of shape [2, num_edges] specifying the edge indices.
    
    """
    residue_index = get_residue_info(pdb_array)
    residue_dm = get_distance_matrix(pdb_array, residue_index, distance_type)
    neighbor_index = get_neighbor_index(residue_dm, num_neighbors)
    source = np.reshape(neighbor_index, (-1, 1)).squeeze(axis=1)
    target = np.repeat(np.arange(residue_index.shape[0]), num_neighbors)
    edge_index = np.stack([source, target])
    return edge_index, residue_dm


def get_edge_coo_data_pre_dm(pdb_array, num_neighbors, dm_file, distance_type='atoms_average'):
    """
    Calculates graph-related data required for the graph neural network in PyTorch Geometric when the distance
    matrix has been pre-calculated.
    
    Args:
        pdb_array (array): 2D array of atom information from PDB file.
        num_neighbors (int): Number of nearest neighbors.
        dm_file (str): Path to pre-calculated distance matrix.
        distance_type (str): Method for calculating residue distance.
    
    Returns:
        An np.ndarray of shape [2, num_edges] specifying the edge indices.
      
    """
    residue_index = get_residue_info(pdb_array)
    residue_dm = np.load(dm_file)
    neighbor_index = get_neighbor_index(residue_dm, num_neighbors)
    source = np.reshape(neighbor_index, (-1, 1)).squeeze(axis=1)
    target = np.repeat(np.arange(residue_index.shape[0]), num_neighbors)
    edge_index = np.stack([source, target])
    return edge_index
