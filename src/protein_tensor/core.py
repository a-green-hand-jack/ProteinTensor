"""
Core classes for protein tensor operations.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, MMCIFParser, MMCIFIO
from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)


class ProteinTensor:
    """
    A class to handle protein structure data conversion between 
    PDB/CIF formats and numpy/torch tensors.
    """
    
    def __init__(
        self, 
        coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
        atom_types: Optional[Union[np.ndarray, torch.Tensor]] = None,
        residue_types: Optional[Union[np.ndarray, torch.Tensor]] = None,
        chain_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        residue_numbers: Optional[Union[np.ndarray, torch.Tensor]] = None,
        structure: Optional[Structure] = None,
    ) -> None:
        """
        Initialize ProteinTensor.
        
        Args:
            coordinates: Atomic coordinates with shape (n_atoms, 3)
            atom_types: Atom type indices with shape (n_atoms,)
            residue_types: Residue type indices with shape (n_residues,)
            chain_ids: Chain identifier indices with shape (n_atoms,)
            residue_numbers: Residue numbers with shape (n_atoms,)
            structure: BioPython Structure object
        """
        self.coordinates = coordinates
        self.atom_types = atom_types
        self.residue_types = residue_types
        self.chain_ids = chain_ids
        self.residue_numbers = residue_numbers
        self._structure = structure
        
        if coordinates is not None:
            self._validate_tensor_shapes()
        
        logger.info(f"ProteinTensor initialized with {self.n_atoms} atoms")
    
    @property
    def n_atoms(self) -> int:
        """Get number of atoms."""
        if self.coordinates is not None:
            return len(self.coordinates)
        return 0
    
    @property
    def n_residues(self) -> int:
        """Get number of residues."""
        if self.residue_numbers is not None:
            return len(np.unique(self.residue_numbers))
        return 0
    
    def _validate_tensor_shapes(self) -> None:
        """Validate that all tensors have consistent shapes."""
        n_atoms = len(self.coordinates)
        
        if self.atom_types is not None and len(self.atom_types) != n_atoms:
            raise ValueError(f"atom_types length {len(self.atom_types)} != n_atoms {n_atoms}")
        
        if self.chain_ids is not None and len(self.chain_ids) != n_atoms:
            raise ValueError(f"chain_ids length {len(self.chain_ids)} != n_atoms {n_atoms}")
        
        if self.residue_numbers is not None and len(self.residue_numbers) != n_atoms:
            raise ValueError(f"residue_numbers length {len(self.residue_numbers)} != n_atoms {n_atoms}")
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Convert all tensors to numpy arrays.
        
        Returns:
            Dictionary containing numpy arrays
        """
        result = {}
        
        if self.coordinates is not None:
            result['coordinates'] = self._tensor_to_numpy(self.coordinates)
        if self.atom_types is not None:
            result['atom_types'] = self._tensor_to_numpy(self.atom_types)
        if self.residue_types is not None:
            result['residue_types'] = self._tensor_to_numpy(self.residue_types)
        if self.chain_ids is not None:
            result['chain_ids'] = self._tensor_to_numpy(self.chain_ids)
        if self.residue_numbers is not None:
            result['residue_numbers'] = self._tensor_to_numpy(self.residue_numbers)
        
        return result
    
    def to_torch(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Convert all arrays to PyTorch tensors.
        
        Args:
            device: Target device for tensors ('cpu', 'cuda', etc.)
            
        Returns:
            Dictionary containing torch tensors
        """
        result = {}
        
        if self.coordinates is not None:
            result['coordinates'] = self._array_to_torch(self.coordinates, device)
        if self.atom_types is not None:
            result['atom_types'] = self._array_to_torch(self.atom_types, device)
        if self.residue_types is not None:
            result['residue_types'] = self._array_to_torch(self.residue_types, device)
        if self.chain_ids is not None:
            result['chain_ids'] = self._array_to_torch(self.chain_ids, device)
        if self.residue_numbers is not None:
            result['residue_numbers'] = self._array_to_torch(self.residue_numbers, device)
        
        return result
    
    def _tensor_to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _array_to_torch(
        self, 
        array: Union[np.ndarray, torch.Tensor], 
        device: Optional[str] = None
    ) -> torch.Tensor:
        """Convert array to torch tensor."""
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    def get_backbone_coords(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Extract backbone atom coordinates (N, CA, C, O).
        
        Returns:
            Backbone coordinates with shape (n_residues, 4, 3) or None
        """
        # This is a placeholder - would need atom type mapping to implement
        logger.warning("get_backbone_coords not fully implemented")
        return None
    
    def center_coordinates(self) -> None:
        """Center coordinates at origin."""
        if self.coordinates is not None:
            if isinstance(self.coordinates, torch.Tensor):
                mean_coord = torch.mean(self.coordinates, dim=0)
                self.coordinates = self.coordinates - mean_coord
            else:
                mean_coord = np.mean(self.coordinates, axis=0)
                self.coordinates = self.coordinates - mean_coord
            
            logger.info("Coordinates centered at origin")
    
    def __repr__(self) -> str:
        return (f"ProteinTensor(n_atoms={self.n_atoms}, "
                f"n_residues={self.n_residues})") 