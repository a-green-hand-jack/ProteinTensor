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


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 