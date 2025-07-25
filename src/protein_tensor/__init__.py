"""
Protein Tensor - Convert protein structures between PDB/CIF formats and numpy/torch tensors.

This package provides functionality to load protein structures from PDB and CIF files
into numpy arrays or PyTorch tensors, and convert them back to structure files.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import ProteinTensor
from .io import load_structure, save_structure
from .utils import get_atom_features, get_residue_features
from .batch_converter import BatchConverter, convert_structures

__all__ = [
    "ProteinTensor",
    "load_structure", 
    "save_structure",
    "get_atom_features",
    "get_residue_features",
    "BatchConverter",
    "convert_structures",
] 