"""
Batch conversion utilities for protein structures.
"""
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import multiprocessing as mp

import numpy as np

from .core import ProteinTensor
from .io import load_structure

logger = logging.getLogger(__name__)


class BatchConverter:
    """
    Batch converter for protein structures to ProteinTensor format.
    
    Supports:
    - Single file or directory processing
    - Recursive directory scanning
    - Parallel processing
    - numpy or torch backend storage
    - Preserving directory structure
    """
    
    def __init__(
        self,
        backend: str = "numpy",
        n_workers: Optional[int] = None,
        preserve_structure: bool = True
    ) -> None:
        """
        Initialize BatchConverter.
        
        Args:
            backend: Storage backend ('numpy' or 'torch')
            n_workers: Number of parallel workers (default: cpu_count() // 2)
            preserve_structure: Whether to preserve directory structure in output
        """
        self.backend = backend.lower()
        if self.backend not in ["numpy", "torch"]:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.n_workers = n_workers or max(1, mp.cpu_count() // 2)
        self.preserve_structure = preserve_structure
        
        logger.info(f"BatchConverter initialized: backend={self.backend}, "
                   f"workers={self.n_workers}, preserve_structure={self.preserve_structure}")
    
    def convert_single(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Convert a single structure file.
        
        Args:
            input_path: Path to input structure file
            output_path: Path to output tensor file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            logger.info(f"Converting {input_path} -> {output_path}")
            
            # Load structure
            protein_tensor = load_structure(input_path)
            
            # Convert to target backend and save
            if self.backend == "numpy":
                tensor_data = protein_tensor.to_numpy()
                # Create metadata
                metadata = np.array({
                    'source_file': str(input_path),
                    'n_atoms': protein_tensor.n_atoms,
                    'n_residues': protein_tensor.n_residues,
                    'backend': 'numpy'
                }, dtype=object)
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save numpy data
                np.savez_compressed(
                    output_path,
                    coordinates=tensor_data['coordinates'],
                    atom_types=tensor_data['atom_types'],
                    residue_types=tensor_data['residue_types'],
                    chain_ids=tensor_data['chain_ids'],
                    residue_numbers=tensor_data['residue_numbers'],
                    metadata=metadata
                )
            else:  # torch
                import torch
                tensor_data = protein_tensor.to_torch()
                
                # Create save data
                save_data = {
                    'coordinates': tensor_data['coordinates'],
                    'atom_types': tensor_data['atom_types'],
                    'residue_types': tensor_data['residue_types'],
                    'chain_ids': tensor_data['chain_ids'],
                    'residue_numbers': tensor_data['residue_numbers'],
                    'metadata': {
                        'source_file': str(input_path),
                        'n_atoms': protein_tensor.n_atoms,
                        'n_residues': protein_tensor.n_residues,
                        'backend': 'torch'
                    }
                }
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save torch data
                torch.save(save_data, output_path)
            
            logger.info(f"Saved tensor data: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            return False
    
    def find_structure_files(
        self,
        input_path: Union[str, Path],
        recursive: bool = True
    ) -> List[Path]:
        """
        Find all structure files in a directory.
        
        Args:
            input_path: Input directory path
            recursive: Whether to search recursively
            
        Returns:
            List of structure file paths
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            return [input_path]
        
        if not input_path.is_dir():
            raise ValueError(f"Input path does not exist: {input_path}")
        
        # Supported extensions
        extensions = {'.pdb', '.ent', '.cif', '.mmcif'}
        
        files = []
        if recursive:
            for ext in extensions:
                files.extend(input_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(input_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} structure files in {input_path}")
        return sorted(files)
    
    def convert_batch(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Convert multiple structure files in batch.
        
        Args:
            input_path: Input file or directory path
            output_dir: Output directory path
            recursive: Whether to search recursively in directories
            
        Returns:
            Dictionary with conversion statistics
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # Find all structure files
        structure_files = self.find_structure_files(input_path, recursive)
        
        if not structure_files:
            logger.warning("No structure files found")
            return {'total': 0, 'success': 0, 'failed': 0, 'failed_files': []}
        
        # Prepare conversion tasks
        tasks = []
        for file_path in structure_files:
            # Determine output path
            if self.preserve_structure and input_path.is_dir():
                # Preserve relative directory structure
                rel_path = file_path.relative_to(input_path)
            else:
                # Flat output structure
                rel_path = file_path.name
            
            # Change extension to appropriate format
            if self.backend == "numpy":
                output_name = Path(rel_path).with_suffix('.npz')
            else:  # torch
                output_name = Path(rel_path).with_suffix('.pt')
            
            output_path = output_dir / output_name
            tasks.append((file_path, output_path))
        
        logger.info(f"Starting batch conversion of {len(tasks)} files using {self.n_workers} workers")
        
        # Process in parallel
        results = {'total': len(tasks), 'success': 0, 'failed': 0, 'failed_files': []}
        
        if self.n_workers == 1:
            # Sequential processing
            for input_file, output_file in tasks:
                success = self.convert_single(input_file, output_file)
                if success:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['failed_files'].append(str(input_file))
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(_convert_worker, str(input_file), str(output_file), self.backend): input_file
                    for input_file, output_file in tasks
                }
                
                # Collect results
                for future in as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            results['success'] += 1
                        else:
                            results['failed'] += 1
                            results['failed_files'].append(str(input_file))
                    except Exception as e:
                        logger.error(f"Worker failed for {input_file}: {e}")
                        results['failed'] += 1
                        results['failed_files'].append(str(input_file))
        
        logger.info(f"Batch conversion completed: {results['success']} success, {results['failed']} failed")
        return results


def _convert_worker(input_path: str, output_path: str, backend: str) -> bool:
    """
    Worker function for parallel processing.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        backend: Storage backend
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Create a new converter instance for this worker
        converter = BatchConverter(backend=backend, n_workers=1)
        return converter.convert_single(input_path, output_path)
    except Exception as e:
        logger.error(f"Worker error for {input_path}: {e}")
        return False


def convert_structures(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    backend: str = "numpy",
    n_workers: Optional[int] = None,
    recursive: bool = True,
    preserve_structure: bool = True
) -> Dict[str, Union[int, List[str]]]:
    """
    Convenience function for batch structure conversion.
    
    Args:
        input_path: Input file or directory path
        output_dir: Output directory path
        backend: Storage backend ('numpy' or 'torch')
        n_workers: Number of parallel workers
        recursive: Whether to search recursively
        preserve_structure: Whether to preserve directory structure
        
    Returns:
        Dictionary with conversion statistics
    """
    converter = BatchConverter(
        backend=backend,
        n_workers=n_workers,
        preserve_structure=preserve_structure
    )
    
    return converter.convert_batch(
        input_path=input_path,
        output_dir=output_dir,
        recursive=recursive
    ) 