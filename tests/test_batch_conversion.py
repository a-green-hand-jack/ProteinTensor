"""
Tests for batch conversion functionality.
"""
import logging
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np

from protein_tensor import convert_structures, BatchConverter, load_structure

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_structure_dir():
    """Create test directory structure with sample files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_structures"
        test_dir.mkdir(exist_ok=True)
        
        # Create subdirectory structure
        (test_dir / "proteins").mkdir(exist_ok=True)
        (test_dir / "ligands").mkdir(exist_ok=True)
        
        # Copy test files to different locations
        # Root directory files
        if Path("9kvc.pdb").exists():
            shutil.copy("9kvc.pdb", test_dir / "protein_1.pdb")
        if Path("9kvc.cif").exists():
            shutil.copy("9kvc.cif", test_dir / "protein_1.cif")
        
        # Subdirectory files
        if Path("9kvc.pdb").exists():
            shutil.copy("9kvc.pdb", test_dir / "proteins" / "complex.pdb")
        if Path("9kvc.cif").exists():
            shutil.copy("9kvc.cif", test_dir / "ligands" / "small_molecule.cif")
        
        logger.info(f"Created test directory structure: {test_dir}")
        yield test_dir
        # Cleanup happens automatically with TemporaryDirectory


class TestBatchConversion:
    """Test batch conversion functionality."""
    
    def test_single_file_conversion(self):
        """Test single file conversion."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "single_conversion_output"
            
            start_time = time.time()
            results = convert_structures(
                input_path=pdb_file,
                output_dir=output_dir,
                backend="numpy",
                n_workers=1
            )
            elapsed_time = time.time() - start_time
            
            # Verify results
            assert results['total'] == 1
            assert results['success'] == 1
            assert results['failed'] == 0
            
            # Verify output files
            output_files = list(output_dir.glob("*.npz"))
            assert len(output_files) == 1
            
            logger.info(f"Single file conversion: {elapsed_time:.2f}s")
    
    def test_batch_conversion_numpy(self, test_structure_dir):
        """Test batch conversion with numpy backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_numpy_output"
            
            start_time = time.time()
            results = convert_structures(
                input_path=test_structure_dir,
                output_dir=output_dir,
                backend="numpy",
                recursive=True,
                preserve_structure=True,
                n_workers=2
            )
            elapsed_time = time.time() - start_time
            
            # Verify results
            assert isinstance(results['total'], int) and results['total'] >= 2  # At least 2 files should be found
            assert results['success'] == results['total']
            assert results['failed'] == 0
            
            # Verify output structure is preserved
            output_files = list(output_dir.rglob("*.npz"))
            assert len(output_files) >= 2
            
            # Check that subdirectories are preserved
            assert any("proteins" in str(f) for f in output_files)
            assert any("ligands" in str(f) for f in output_files)
            
            logger.info(f"Numpy batch conversion: {elapsed_time:.2f}s, {len(output_files)} files")
    
    def test_batch_conversion_torch(self, test_structure_dir):
        """Test batch conversion with torch backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_torch_output"
            
            start_time = time.time()
            results = convert_structures(
                input_path=test_structure_dir,
                output_dir=output_dir,
                backend="torch",
                recursive=True,
                preserve_structure=False,  # Flat output
                n_workers=1
            )
            elapsed_time = time.time() - start_time
            
            # Verify results
            assert isinstance(results['total'], int) and results['total'] >= 2
            assert results['success'] == results['total']
            assert results['failed'] == 0
            
            # Verify flat output structure
            output_files = list(output_dir.glob("*.pt"))
            assert len(output_files) >= 2
            
            # Check that files are in root directory (flat structure)
            for file in output_files:
                assert file.parent == output_dir
            
            logger.info(f"Torch batch conversion: {elapsed_time:.2f}s, {len(output_files)} files")
    
    def test_data_consistency_numpy(self):
        """Test data consistency between original and converted data (numpy)."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        # Load original structure
        original_protein = load_structure(pdb_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "consistency_test"
            
            # Convert using batch converter
            results = convert_structures(
                input_path=pdb_file,
                output_dir=output_dir,
                backend="numpy",
                n_workers=1
            )
            
            assert results['success'] == 1
            
            # Load converted data
            converted_files = list(output_dir.glob("*.npz"))
            assert len(converted_files) == 1
            
            data = np.load(converted_files[0], allow_pickle=True)
            
            # Verify data consistency
            assert data['coordinates'].shape[0] == original_protein.n_atoms
            assert len(data['atom_types']) == original_protein.n_atoms
            assert len(data['residue_types']) == original_protein.n_atoms
            assert len(data['chain_ids']) == original_protein.n_atoms
            assert len(data['residue_numbers']) == original_protein.n_atoms
            
            # Check metadata
            metadata = data['metadata'].item()
            assert metadata['n_atoms'] == original_protein.n_atoms
            assert metadata['n_residues'] == original_protein.n_residues
            assert metadata['backend'] == 'numpy'
            assert str(pdb_file) in metadata['source_file']
            
            # Compare coordinates (should be identical within floating point precision)
            original_coords = original_protein.to_numpy()['coordinates']
            np.testing.assert_allclose(
                data['coordinates'], 
                original_coords, 
                rtol=1e-6, 
                atol=1e-6,
                err_msg="Coordinates should be identical between original and converted data"
            )
            
            logger.info(f"Data consistency verified: {data['coordinates'].shape[0]} atoms")
    
    def test_data_consistency_torch(self):
        """Test data consistency between original and converted data (torch)."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Load original structure
        original_protein = load_structure(pdb_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "torch_consistency_test"
            
            # Convert using batch converter
            results = convert_structures(
                input_path=pdb_file,
                output_dir=output_dir,
                backend="torch",
                n_workers=1
            )
            
            assert results['success'] == 1
            
            # Load converted data
            converted_files = list(output_dir.glob("*.pt"))
            assert len(converted_files) == 1
            
            data = torch.load(converted_files[0])
            
            # Verify data consistency
            assert data['coordinates'].shape[0] == original_protein.n_atoms
            assert len(data['atom_types']) == original_protein.n_atoms
            assert len(data['residue_types']) == original_protein.n_atoms
            assert len(data['chain_ids']) == original_protein.n_atoms
            assert len(data['residue_numbers']) == original_protein.n_atoms
            
            # Check metadata
            metadata = data['metadata']
            assert metadata['n_atoms'] == original_protein.n_atoms
            assert metadata['n_residues'] == original_protein.n_residues
            assert metadata['backend'] == 'torch'
            assert str(pdb_file) in metadata['source_file']
            
            # Compare coordinates (convert to numpy for comparison)
            original_coords = original_protein.to_numpy()['coordinates']
            converted_coords = data['coordinates'].cpu().numpy()
            np.testing.assert_allclose(
                converted_coords, 
                original_coords, 
                rtol=1e-6, 
                atol=1e-6,
                err_msg="Coordinates should be identical between original and converted torch data"
            )
            
            logger.info(f"Torch data consistency verified: {data['coordinates'].shape[0]} atoms")
    
    def test_roundtrip_consistency(self):
        """Test roundtrip consistency: original -> tensor -> structure -> tensor."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        # Load original structure
        original_protein = load_structure(pdb_file)
        original_coords = original_protein.to_numpy()['coordinates']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Convert to tensor format
            output_dir = Path(temp_dir) / "roundtrip_test"
            results = convert_structures(
                input_path=pdb_file,
                output_dir=output_dir,
                backend="numpy",
                n_workers=1
            )
            assert results['success'] == 1
            
            # Step 2: Load tensor data and create new ProteinTensor
            converted_files = list(output_dir.glob("*.npz"))
            data = np.load(converted_files[0], allow_pickle=True)
            
            from protein_tensor import ProteinTensor
            reconstructed_protein = ProteinTensor(
                coordinates=data['coordinates'],
                atom_types=data['atom_types'],
                residue_types=data['residue_types'],
                chain_ids=data['chain_ids'],
                residue_numbers=data['residue_numbers']
            )
            
            # Step 3: Save reconstructed structure and reload
            reconstructed_pdb = output_dir / "reconstructed.pdb"
            from protein_tensor import save_structure
            save_structure(reconstructed_protein, reconstructed_pdb)
            
            reloaded_protein = load_structure(reconstructed_pdb)
            reloaded_coords = reloaded_protein.to_numpy()['coordinates']
            
            # Verify roundtrip consistency
            assert original_protein.n_atoms == reloaded_protein.n_atoms
            
            # Coordinates should be very close (allowing for floating point precision and format conversion)
            np.testing.assert_allclose(
                original_coords,
                reloaded_coords,
                rtol=1e-3,  # More relaxed tolerance for roundtrip
                atol=1e-3,
                err_msg="Roundtrip coordinates should be consistent"
            )
            
            logger.info(f"Roundtrip consistency verified: {original_protein.n_atoms} atoms")
    
    def test_converter_class_direct_usage(self):
        """Test BatchConverter class direct usage."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        # Create converter instance
        converter = BatchConverter(
            backend="numpy",
            n_workers=1,
            preserve_structure=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test.npz"
            
            # Test single file conversion
            success = converter.convert_single(pdb_file, output_file)
            assert success
            assert output_file.exists()
            
            # Verify converted data
            data = np.load(output_file, allow_pickle=True)
            assert 'coordinates' in data
            assert 'metadata' in data
            
            metadata = data['metadata'].item()
            assert metadata['n_atoms'] > 0
            assert metadata['backend'] == 'numpy'
            
            logger.info(f"Converter class test: {output_file.stat().st_size // 1024} KB")
    
    def test_cross_format_consistency(self):
        """Test consistency between PDB and CIF conversions."""
        pdb_file = Path("9kvc.pdb") 
        cif_file = Path("9kvc.cif")
        
        if not (pdb_file.exists() and cif_file.exists()):
            pytest.skip("Both PDB and CIF test files required")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "cross_format_test"
            
            # Convert both files
            pdb_results = convert_structures(pdb_file, output_dir / "pdb", backend="numpy")
            cif_results = convert_structures(cif_file, output_dir / "cif", backend="numpy")
            
            assert pdb_results['success'] == 1
            assert cif_results['success'] == 1
            
            # Load converted data
            pdb_data = np.load(list((output_dir / "pdb").glob("*.npz"))[0], allow_pickle=True)
            cif_data = np.load(list((output_dir / "cif").glob("*.npz"))[0], allow_pickle=True)
            
            # Compare metadata
            pdb_meta = pdb_data['metadata'].item()
            cif_meta = cif_data['metadata'].item()
            
            # Should have same number of atoms (allowing small differences due to format variations)
            atom_diff = abs(pdb_meta['n_atoms'] - cif_meta['n_atoms'])
            assert atom_diff <= 100, f"Large atom count difference: {atom_diff}"
            
            residue_diff = abs(pdb_meta['n_residues'] - cif_meta['n_residues']) 
            assert residue_diff <= 10, f"Large residue count difference: {residue_diff}"
            
            logger.info(f"Cross-format consistency: PDB={pdb_meta['n_atoms']} atoms, "
                       f"CIF={cif_meta['n_atoms']} atoms, diff={atom_diff}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 