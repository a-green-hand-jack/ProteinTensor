"""
Utility functions and constants for protein tensor operations.
"""
import logging
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Standard amino acid residue types
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Common atom types in proteins
COMMON_ATOM_TYPES = [
    'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
    'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'ND1', 'ND2',
    'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2',
    'OG', 'OG1', 'OH', 'SD', 'SG'
]

# Create mappings from names to indices
RESIDUE_TYPE_MAP: Dict[str, int] = {res: i for i, res in enumerate(STANDARD_AMINO_ACIDS)}
ATOM_TYPE_MAP: Dict[str, int] = {atom: i for i, atom in enumerate(COMMON_ATOM_TYPES)}

# Reverse mappings
RESIDUE_INDEX_MAP: Dict[int, str] = {i: res for res, i in RESIDUE_TYPE_MAP.items()}
ATOM_INDEX_MAP: Dict[int, str] = {i: atom for atom, i in ATOM_TYPE_MAP.items()}

# Atomic properties (atomic numbers)
ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15
}


def get_atom_features(atom_name: str) -> Dict[str, Union[int, float]]:
    """
    Get atomic features for a given atom name.
    
    Args:
        atom_name: Name of the atom (e.g., 'CA', 'N', 'O')
        
    Returns:
        Dictionary containing atomic features
    """
    features = {}
    
    # Get atom type index
    features['type_index'] = ATOM_TYPE_MAP.get(atom_name, len(ATOM_TYPE_MAP))
    
    # Infer element from atom name (simplified)
    element = _infer_element(atom_name)
    features['atomic_number'] = ATOMIC_NUMBERS.get(element, 0)
    features['element'] = element
    
    # Add more features as needed
    features['is_backbone'] = atom_name in ['N', 'CA', 'C', 'O']
    features['is_sidechain'] = not features['is_backbone']
    
    return features


def get_residue_features(residue_name: str) -> Dict[str, Union[int, float, bool]]:
    """
    Get residue features for a given residue name.
    
    Args:
        residue_name: Three-letter residue code (e.g., 'ALA', 'GLY')
        
    Returns:
        Dictionary containing residue features
    """
    features = {}
    
    # Get residue type index
    features['type_index'] = RESIDUE_TYPE_MAP.get(residue_name, len(RESIDUE_TYPE_MAP))
    
    # Add biochemical properties
    features['is_polar'] = residue_name in ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS']
    features['is_charged'] = residue_name in ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']
    features['is_positive'] = residue_name in ['ARG', 'LYS', 'HIS']
    features['is_negative'] = residue_name in ['ASP', 'GLU']
    features['is_aromatic'] = residue_name in ['PHE', 'TRP', 'TYR', 'HIS']
    features['is_hydrophobic'] = residue_name in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
    features['is_small'] = residue_name in ['ALA', 'GLY', 'SER', 'THR', 'CYS']
    
    # Molecular properties (approximate)
    mw_map = {
        'ALA': 89.1, 'ARG': 174.2, 'ASN': 132.1, 'ASP': 133.1, 'CYS': 121.0,
        'GLN': 146.1, 'GLU': 147.1, 'GLY': 75.1, 'HIS': 155.2, 'ILE': 131.2,
        'LEU': 131.2, 'LYS': 146.2, 'MET': 149.2, 'PHE': 165.2, 'PRO': 115.1,
        'SER': 105.1, 'THR': 119.1, 'TRP': 204.2, 'TYR': 181.2, 'VAL': 117.1
    }
    features['molecular_weight'] = mw_map.get(residue_name, 0.0)
    
    return features


def _infer_element(atom_name: str) -> str:
    """
    Infer chemical element from atom name.
    
    Args:
        atom_name: Atom name (e.g., 'CA', 'N', 'O')
        
    Returns:
        Chemical element symbol
    """
    # Simple heuristic based on first character
    first_char = atom_name[0].upper()
    
    if first_char == 'C':
        return 'C'
    elif first_char == 'N':
        return 'N'
    elif first_char == 'O':
        return 'O'
    elif first_char == 'S':
        return 'S'
    elif first_char == 'P':
        return 'P'
    elif first_char == 'H':
        return 'H'
    else:
        # Default to carbon for unknown
        return 'C'


def tensor_info(tensor: Union[np.ndarray, 'torch.Tensor']) -> str:
    """
    Get information about a tensor.
    
    Args:
        tensor: Numpy array or PyTorch tensor
        
    Returns:
        String description of tensor properties
    """
    try:
        import torch
        is_torch = isinstance(tensor, torch.Tensor)
    except ImportError:
        is_torch = False
    
    if is_torch:
        return (f"torch.Tensor(shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, device={tensor.device})")
    else:
        return (f"numpy.ndarray(shape={tensor.shape}, "
                f"dtype={tensor.dtype})")


def center_coordinates(coords: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Center coordinates at the origin.
    
    Args:
        coords: Coordinate array with shape (..., 3)
        
    Returns:
        Centered coordinates
    """
    try:
        import torch
        if isinstance(coords, torch.Tensor):
            mean_coord = torch.mean(coords, dim=0)
            return coords - mean_coord
    except ImportError:
        pass
    
    # Numpy case
    mean_coord = np.mean(coords, axis=0)
    return coords - mean_coord


def calculate_distances(
    coords1: Union[np.ndarray, 'torch.Tensor'],
    coords2: Optional[Union[np.ndarray, 'torch.Tensor']] = None
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Calculate pairwise distances between coordinates.
    
    Args:
        coords1: First set of coordinates with shape (N, 3)
        coords2: Second set of coordinates with shape (M, 3). 
                If None, calculate distances within coords1
        
    Returns:
        Distance matrix with shape (N, M) or (N, N)
    """
    if coords2 is None:
        coords2 = coords1
    
    try:
        import torch
        if isinstance(coords1, torch.Tensor):
            # PyTorch implementation
            diff = coords1.unsqueeze(1) - coords2.unsqueeze(0)  # (N, M, 3)
            return torch.sqrt(torch.sum(diff**2, dim=2))  # (N, M)
    except ImportError:
        pass
    
    # Numpy implementation
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]  # (N, M, 3)
    return np.sqrt(np.sum(diff**2, axis=2))  # (N, M) 