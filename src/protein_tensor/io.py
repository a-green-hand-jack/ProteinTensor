"""
Input/Output operations for protein structures.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, MMCIFParser, MMCIFIO
from Bio.PDB.Structure import Structure

from .core import ProteinTensor
from .utils import (
    ATOM_TYPE_MAP, 
    RESIDUE_TYPE_MAP,
    get_atom_features,
    get_residue_features
)

logger = logging.getLogger(__name__)


def load_structure(
    filepath: Union[str, Path], 
    format_type: Optional[str] = None
) -> ProteinTensor:
    """
    Load protein structure from PDB or CIF file.
    
    Args:
        filepath: Path to structure file
        format_type: File format ('pdb' or 'cif'). If None, inferred from extension
        
    Returns:
        ProteinTensor object containing structure data
    """
    filepath = Path(filepath)
    
    if format_type is None:
        format_type = _infer_format(filepath)
    
    logger.info(f"Loading {format_type.upper()} file: {filepath}")
    
    # Parse structure
    if format_type.lower() == 'pdb':
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(filepath))
    elif format_type.lower() == 'cif':
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', str(filepath))
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Extract tensor data
    return _structure_to_tensor(structure)


def save_structure(
    protein_tensor: ProteinTensor,
    filepath: Union[str, Path],
    format_type: Optional[str] = None
) -> None:
    """
    Save ProteinTensor to PDB or CIF file.
    
    Args:
        protein_tensor: ProteinTensor object to save
        filepath: Output file path
        format_type: File format ('pdb' or 'cif'). If None, inferred from extension
    """
    filepath = Path(filepath)
    
    if format_type is None:
        format_type = _infer_format(filepath)
    
    logger.info(f"Saving {format_type.upper()} file: {filepath}")
    
    # Convert tensor back to BioPython structure
    structure = _tensor_to_structure(protein_tensor)
    
    # Save structure
    if format_type.lower() == 'pdb':
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(filepath))
    elif format_type.lower() == 'cif':
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(filepath))
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def _infer_format(filepath: Path) -> str:
    """Infer file format from extension."""
    suffix = filepath.suffix.lower()
    if suffix in ['.pdb', '.ent']:
        return 'pdb'
    elif suffix in ['.cif', '.mmcif']:
        return 'cif'
    else:
        raise ValueError(f"Cannot infer format from extension: {suffix}")


def _structure_to_tensor(structure: Structure) -> ProteinTensor:
    """
    Convert BioPython Structure to ProteinTensor.
    
    Args:
        structure: BioPython Structure object
        
    Returns:
        ProteinTensor object
    """
    coordinates = []
    atom_types = []
    residue_types = []
    chain_ids = []
    residue_numbers = []
    
    chain_id_map = {}
    chain_counter = 0
    
    for model in structure:
        for chain in model:
            # Map chain IDs to integers
            if chain.id not in chain_id_map:
                chain_id_map[chain.id] = chain_counter
                chain_counter += 1
            
            chain_idx = chain_id_map[chain.id]
            
            for residue in chain:
                # Skip HETATM records for now
                if residue.id[0] != ' ':
                    continue
                
                res_name = residue.get_resname()
                res_num = residue.id[1]
                
                # Get residue type index
                res_type_idx = RESIDUE_TYPE_MAP.get(res_name, len(RESIDUE_TYPE_MAP))
                
                for atom in residue:
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    
                    # Get atom type index
                    atom_type_idx = ATOM_TYPE_MAP.get(atom_name, len(ATOM_TYPE_MAP))
                    
                    coordinates.append(coord)
                    atom_types.append(atom_type_idx)
                    residue_types.append(res_type_idx)
                    chain_ids.append(chain_idx)
                    residue_numbers.append(res_num)
    
    # Convert to numpy arrays
    coordinates = np.array(coordinates, dtype=np.float32)
    atom_types = np.array(atom_types, dtype=np.int32)
    residue_types = np.array(residue_types, dtype=np.int32)
    chain_ids = np.array(chain_ids, dtype=np.int32)
    residue_numbers = np.array(residue_numbers, dtype=np.int32)
    
    logger.info(f"Extracted {len(coordinates)} atoms from structure")
    
    return ProteinTensor(
        coordinates=coordinates,
        atom_types=atom_types,
        residue_types=residue_types,
        chain_ids=chain_ids,
        residue_numbers=residue_numbers,
        structure=structure
    )


def _tensor_to_structure(protein_tensor: ProteinTensor) -> Structure:
    """
    Convert ProteinTensor back to BioPython Structure.
    
    Args:
        protein_tensor: ProteinTensor object
        
    Returns:
        BioPython Structure object
    """
    # This is a placeholder implementation
    # In practice, this would require reverse mapping from indices to names
    # and proper structure building
    logger.warning("_tensor_to_structure not fully implemented")
    
    if protein_tensor._structure is not None:
        return protein_tensor._structure
    
    # Create a minimal structure for now
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    
    structure = Structure('converted')
    model = Model(0)
    chain = Chain('A')
    
    # This is a very basic implementation
    # Would need proper atom/residue name mapping
    structure.add(model)
    model.add(chain)
    
    return structure 