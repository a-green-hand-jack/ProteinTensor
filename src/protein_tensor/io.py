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
    Convert ProteinTensor back to BioPython Structure from tensor data.
    
    Args:
        protein_tensor: ProteinTensor object
        
    Returns:
        BioPython Structure object rebuilt from tensor data
    """
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    
    if protein_tensor.coordinates is None:
        raise ValueError("No coordinates available to build structure")
    
    logger.info("Rebuilding structure from tensor data")
    
    # Get numpy arrays from tensor data
    coords = protein_tensor._tensor_to_numpy(protein_tensor.coordinates)
    atom_types = protein_tensor._tensor_to_numpy(protein_tensor.atom_types) if protein_tensor.atom_types is not None else None
    residue_types = protein_tensor._tensor_to_numpy(protein_tensor.residue_types) if protein_tensor.residue_types is not None else None
    chain_ids = protein_tensor._tensor_to_numpy(protein_tensor.chain_ids) if protein_tensor.chain_ids is not None else None
    residue_numbers = protein_tensor._tensor_to_numpy(protein_tensor.residue_numbers) if protein_tensor.residue_numbers is not None else None
    
    # Create reverse mappings
    from .utils import ATOM_INDEX_MAP, RESIDUE_INDEX_MAP
    
    # Create new structure
    structure = Structure('rebuilt_from_tensor')
    model = Model(0)
    structure.add(model)
    
    # Group atoms by chain and residue
    current_chain_id = None
    current_residue_num = None
    current_chain = None
    current_residue = None
    atom_serial = 1
    
    for i in range(len(coords)):
        # Get chain ID
        if chain_ids is not None:
            chain_id_idx = int(chain_ids[i])
            chain_id_str = chr(ord('A') + chain_id_idx) if chain_id_idx < 26 else f"A{chain_id_idx}"
        else:
            chain_id_str = 'A'
        
        # Get residue number
        if residue_numbers is not None:
            res_num = int(residue_numbers[i])
        else:
            res_num = 1
        
        # Create new chain if needed
        if current_chain_id != chain_id_str:
            current_chain = Chain(chain_id_str)
            model.add(current_chain)
            current_chain_id = chain_id_str
            current_residue_num = None
        
        # Create new residue if needed
        if current_residue_num != res_num:
            # Get residue name from index
            if residue_types is not None:
                res_type_idx = int(residue_types[i])
                res_name = RESIDUE_INDEX_MAP.get(res_type_idx, 'UNK')
            else:
                res_name = 'UNK'
            
            current_residue = Residue((' ', res_num, ' '), res_name, ' ')
            current_chain.add(current_residue)
            current_residue_num = res_num
        
        # Get atom name from index
        if atom_types is not None:
            atom_type_idx = int(atom_types[i])
            atom_name = ATOM_INDEX_MAP.get(atom_type_idx, 'X')
        else:
            atom_name = 'CA'  # Default to CA
        
        # Create atom
        coord = coords[i].astype(float)
        atom = Atom(
            name=atom_name,
            coord=coord,
            bfactor=0.0,
            occupancy=1.0,
            altloc=' ',
            fullname=f" {atom_name:<3}",
            serial_number=atom_serial
        )
        
        current_residue.add(atom)
        atom_serial += 1
    
    logger.info(f"Rebuilt structure with {len(coords)} atoms")
    return structure 