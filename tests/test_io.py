"""
Tests for protein_tensor I/O operations.
"""
import logging
import tempfile
from pathlib import Path

import pytest
import numpy as np

from protein_tensor import load_structure, save_structure, ProteinTensor

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIO:
    """Test input/output operations."""
    
    def test_load_pdb_structure(self):
        """Test loading PDB structure."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        protein_tensor = load_structure(pdb_file)
        
        assert isinstance(protein_tensor, ProteinTensor)
        assert protein_tensor.n_atoms > 0
        assert protein_tensor.coordinates is not None
        assert protein_tensor.coordinates.shape[1] == 3  # Should have x, y, z
        
        logger.info(f"Loaded PDB with {protein_tensor.n_atoms} atoms")
    
    def test_load_cif_structure(self):
        """Test loading CIF structure."""
        cif_file = Path("9kvc.cif")
        if not cif_file.exists():
            pytest.skip("Test CIF file not found")
        
        protein_tensor = load_structure(cif_file)
        
        assert isinstance(protein_tensor, ProteinTensor)
        assert protein_tensor.n_atoms > 0
        assert protein_tensor.coordinates is not None
        assert protein_tensor.coordinates.shape[1] == 3
        
        logger.info(f"Loaded CIF with {protein_tensor.n_atoms} atoms")
    
    def test_tensor_conversions(self):
        """Test numpy and torch tensor conversions."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        protein_tensor = load_structure(pdb_file)
        
        # Test numpy conversion
        numpy_data = protein_tensor.to_numpy()
        assert 'coordinates' in numpy_data
        assert isinstance(numpy_data['coordinates'], np.ndarray)
        
        # Test torch conversion if available
        try:
            import torch
            torch_data = protein_tensor.to_torch()
            assert 'coordinates' in torch_data
            assert isinstance(torch_data['coordinates'], torch.Tensor)
            logger.info("PyTorch conversion successful")
        except ImportError:
            logger.warning("PyTorch not available, skipping torch conversion test")
    
    def test_save_structure(self):
        """Test saving structure."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        # Load structure
        protein_tensor = load_structure(pdb_file)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            save_structure(protein_tensor, tmp_path)
            assert tmp_path.exists()
            
            # Try to load it back
            reloaded = load_structure(tmp_path)
            assert reloaded.n_atoms > 0
            
            logger.info(f"Successfully saved and reloaded structure")
        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_center_coordinates(self):
        """Test coordinate centering."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        protein_tensor = load_structure(pdb_file)
        
        # Store original coordinates
        if protein_tensor.coordinates is not None:
            # Convert to numpy for easier handling
            original_coords = np.array(protein_tensor.coordinates)
        else:
            pytest.skip("No coordinates found")
        
        # Center coordinates
        protein_tensor.center_coordinates()
        
        # Check that coordinates changed
        if protein_tensor.coordinates is not None:
            assert not np.allclose(original_coords, protein_tensor.coordinates)
            
            # Check that mean is close to zero
            mean_coord = np.mean(protein_tensor.coordinates, axis=0)
            assert np.allclose(mean_coord, [0, 0, 0], atol=1e-4)
        
        logger.info("Coordinate centering test passed")
    
    def test_protein_tensor_properties(self):
        """Test ProteinTensor properties and methods."""
        pdb_file = Path("9kvc.pdb")
        if not pdb_file.exists():
            pytest.skip("Test PDB file not found")
        
        protein_tensor = load_structure(pdb_file)
        
        # Test properties
        assert protein_tensor.n_atoms > 0
        assert protein_tensor.n_residues > 0
        
        # Test string representation
        repr_str = repr(protein_tensor)
        assert "ProteinTensor" in repr_str
        assert str(protein_tensor.n_atoms) in repr_str
        
        logger.info(f"ProteinTensor properties: {repr_str}")


    def test_complete_conversion_workflow(self):
        """Test complete conversion workflow: PDB/CIF -> numpy/tensor -> PDB/CIF."""
        import tempfile
        import shutil
        from pathlib import Path
        
        # Create test output directory
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Test 1: PDB -> numpy -> PDB, PDB -> tensor -> CIF
        pdb_file = Path("9kvc.pdb")
        if pdb_file.exists():
            logger.info("Testing PDB conversion workflow")
            
            # Load PDB
            protein_from_pdb = load_structure(pdb_file)
            logger.info(f"Loaded PDB: {protein_from_pdb.n_atoms} atoms, {protein_from_pdb.n_residues} residues")
            
            # Convert to numpy and save as PDB (without original structure)
            numpy_data = protein_from_pdb.to_numpy()
            pdb_from_numpy = ProteinTensor(
                coordinates=numpy_data['coordinates'],
                atom_types=numpy_data['atom_types'],
                residue_types=numpy_data['residue_types'],
                chain_ids=numpy_data['chain_ids'],
                residue_numbers=numpy_data['residue_numbers']
                # Intentionally NOT passing structure to test tensor->structure conversion
            )
            pdb_output_path = output_dir / "from_pdb_via_numpy.pdb"
            save_structure(pdb_from_numpy, pdb_output_path)
            logger.info(f"Saved PDB via numpy: {pdb_output_path}")
            
            # Convert to torch tensor if available and save as CIF
            try:
                torch_data = protein_from_pdb.to_torch()
                pdb_from_torch = ProteinTensor(
                    coordinates=torch_data['coordinates'],
                    atom_types=torch_data['atom_types'],
                    residue_types=torch_data['residue_types'],
                    chain_ids=torch_data['chain_ids'],
                    residue_numbers=torch_data['residue_numbers']
                    # Intentionally NOT passing structure to test tensor->structure conversion
                )
                cif_output_path = output_dir / "from_pdb_via_torch.cif"
                save_structure(pdb_from_torch, cif_output_path)
                logger.info(f"Saved CIF via torch: {cif_output_path}")
                
                # Verify the torch tensors have correct properties
                assert torch_data['coordinates'].shape == numpy_data['coordinates'].shape
                logger.info("Torch tensor shapes verified")
                
            except ImportError:
                logger.warning("PyTorch not available, skipping torch conversion test")
            
            # Verify saved files exist and are non-empty
            assert pdb_output_path.exists() and pdb_output_path.stat().st_size > 0
            
        else:
            pytest.skip("Test PDB file not found")
        
        # Test 2: CIF -> numpy -> PDB, CIF -> tensor -> CIF
        cif_file = Path("9kvc.cif")
        if cif_file.exists():
            logger.info("Testing CIF conversion workflow")
            
            # Load CIF
            protein_from_cif = load_structure(cif_file)
            logger.info(f"Loaded CIF: {protein_from_cif.n_atoms} atoms, {protein_from_cif.n_residues} residues")
            
            # Convert to numpy and save as PDB
            numpy_data_cif = protein_from_cif.to_numpy()
            cif_from_numpy = ProteinTensor(
                coordinates=numpy_data_cif['coordinates'],
                atom_types=numpy_data_cif['atom_types'],
                residue_types=numpy_data_cif['residue_types'],
                chain_ids=numpy_data_cif['chain_ids'],
                residue_numbers=numpy_data_cif['residue_numbers']
                # Intentionally NOT passing structure to test tensor->structure conversion
            )
            pdb_from_cif_path = output_dir / "from_cif_via_numpy.pdb"
            save_structure(cif_from_numpy, pdb_from_cif_path)
            logger.info(f"Saved PDB from CIF via numpy: {pdb_from_cif_path}")
            
            # Convert to torch tensor if available and save as CIF
            try:
                torch_data_cif = protein_from_cif.to_torch()
                cif_from_torch = ProteinTensor(
                    coordinates=torch_data_cif['coordinates'],
                    atom_types=torch_data_cif['atom_types'],
                    residue_types=torch_data_cif['residue_types'],
                    chain_ids=torch_data_cif['chain_ids'],
                    residue_numbers=torch_data_cif['residue_numbers']
                    # Intentionally NOT passing structure to test tensor->structure conversion
                )
                cif_from_cif_path = output_dir / "from_cif_via_torch.cif"
                save_structure(cif_from_torch, cif_from_cif_path)
                logger.info(f"Saved CIF from CIF via torch: {cif_from_cif_path}")
                
                # Verify the torch tensors have correct properties
                assert torch_data_cif['coordinates'].shape == numpy_data_cif['coordinates'].shape
                logger.info("CIF torch tensor shapes verified")
                
            except ImportError:
                logger.warning("PyTorch not available, skipping CIF torch conversion test")
            
            # Verify saved files exist and are non-empty
            assert pdb_from_cif_path.exists() and pdb_from_cif_path.stat().st_size > 0
            
        else:
            pytest.skip("Test CIF file not found")
        
        # Cross-validation: compare data consistency
        if pdb_file.exists() and cif_file.exists():
            logger.info("Cross-validating PDB and CIF data consistency")
            
            # Both should have similar atom counts (allowing for small differences due to format differences)
            atom_count_diff = abs(protein_from_pdb.n_atoms - protein_from_cif.n_atoms)
            residue_count_diff = abs(protein_from_pdb.n_residues - protein_from_cif.n_residues)
            
            # Allow some tolerance for format differences
            assert atom_count_diff <= 100, f"Large atom count difference: {atom_count_diff}"
            assert residue_count_diff <= 10, f"Large residue count difference: {residue_count_diff}"
            
            logger.info(f"Cross-validation passed: atom diff={atom_count_diff}, residue diff={residue_count_diff}")
        
        logger.info("Complete conversion workflow test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 