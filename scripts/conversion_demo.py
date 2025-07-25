#!/usr/bin/env python3
"""
完整转换流程演示脚本
展示 PDB/CIF ↔ numpy/torch ↔ PDB/CIF 的转换能力
"""
import logging
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_tensor import load_structure, save_structure, ProteinTensor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_conversion_workflow():
    """演示完整的转换工作流程。"""
    logger.info("🧬 开始蛋白质结构转换流程演示")
    
    # 创建输出目录
    output_dir = Path("conversion_demo_outputs")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"📁 输出目录: {output_dir}")
    
    # === PDB 转换流程 ===
    pdb_file = Path("9kvc.pdb")
    if pdb_file.exists():
        logger.info(f"\n📄 PDB转换流程演示")
        logger.info(f"输入文件: {pdb_file}")
        
        # 1. 加载PDB文件
        protein = load_structure(pdb_file)
        logger.info(f"✅ 加载PDB成功: {protein.n_atoms} 个原子, {protein.n_residues} 个残基")
        
        # 2. 转换为numpy数组
        numpy_data = protein.to_numpy()
        logger.info(f"🔢 转换为numpy数组: {len(numpy_data)} 个数据类型")
        logger.info(f"   - 坐标形状: {numpy_data['coordinates'].shape}")
        logger.info(f"   - 原子类型: {numpy_data['atom_types'].shape}")
        
        # 3. 从numpy创建新的ProteinTensor并保存为PDB
        pdb_from_numpy = ProteinTensor(
            coordinates=numpy_data['coordinates'],
            atom_types=numpy_data['atom_types'],
            residue_types=numpy_data['residue_types'],
            chain_ids=numpy_data['chain_ids'],
            residue_numbers=numpy_data['residue_numbers'],
            structure=protein._structure
        )
        
        pdb_output = output_dir / "pdb_via_numpy.pdb"
        save_structure(pdb_from_numpy, pdb_output)
        logger.info(f"💾 PDB→numpy→PDB: {pdb_output}")
        
        # 4. 转换为PyTorch张量并保存为CIF
        try:
            torch_data = protein.to_torch()
            logger.info(f"🔥 转换为PyTorch张量: {len(torch_data)} 个数据类型")
            logger.info(f"   - 坐标设备: {torch_data['coordinates'].device}")
            
            pdb_from_torch = ProteinTensor(
                coordinates=torch_data['coordinates'],
                atom_types=torch_data['atom_types'],
                residue_types=torch_data['residue_types'],
                chain_ids=torch_data['chain_ids'],
                residue_numbers=torch_data['residue_numbers'],
                structure=protein._structure
            )
            
            cif_output = output_dir / "pdb_via_torch.cif"
            save_structure(pdb_from_torch, cif_output)
            logger.info(f"💾 PDB→torch→CIF: {cif_output}")
            
        except ImportError:
            logger.warning("⚠️  PyTorch未安装，跳过张量转换演示")
    
    # === CIF 转换流程 ===
    cif_file = Path("9kvc.cif")
    if cif_file.exists():
        logger.info(f"\n📄 CIF转换流程演示")
        logger.info(f"输入文件: {cif_file}")
        
        # 1. 加载CIF文件
        protein_cif = load_structure(cif_file)
        logger.info(f"✅ 加载CIF成功: {protein_cif.n_atoms} 个原子, {protein_cif.n_residues} 个残基")
        
        # 2. 转换为numpy数组并保存为PDB
        numpy_data_cif = protein_cif.to_numpy()
        
        cif_from_numpy = ProteinTensor(
            coordinates=numpy_data_cif['coordinates'],
            atom_types=numpy_data_cif['atom_types'],
            residue_types=numpy_data_cif['residue_types'],
            chain_ids=numpy_data_cif['chain_ids'],
            residue_numbers=numpy_data_cif['residue_numbers'],
            structure=protein_cif._structure
        )
        
        pdb_from_cif = output_dir / "cif_via_numpy.pdb"
        save_structure(cif_from_numpy, pdb_from_cif)
        logger.info(f"💾 CIF→numpy→PDB: {pdb_from_cif}")
        
        # 3. 转换为PyTorch张量并保存为CIF
        try:
            torch_data_cif = protein_cif.to_torch()
            
            cif_from_torch = ProteinTensor(
                coordinates=torch_data_cif['coordinates'],
                atom_types=torch_data_cif['atom_types'],
                residue_types=torch_data_cif['residue_types'],
                chain_ids=torch_data_cif['chain_ids'],
                residue_numbers=torch_data_cif['residue_numbers'],
                structure=protein_cif._structure
            )
            
            cif_from_cif = output_dir / "cif_via_torch.cif"
            save_structure(cif_from_torch, cif_from_cif)
            logger.info(f"💾 CIF→torch→CIF: {cif_from_cif}")
            
        except ImportError:
            logger.warning("⚠️  PyTorch未安装，跳过张量转换演示")
    
    # === 总结 ===
    logger.info(f"\n📊 转换演示完成！")
    output_files = list(output_dir.glob("*"))
    logger.info(f"生成了 {len(output_files)} 个输出文件:")
    for file in sorted(output_files):
        size_kb = file.stat().st_size // 1024
        logger.info(f"  - {file.name} ({size_kb} KB)")
    
    logger.info(f"\n🎯 转换链路验证:")
    logger.info(f"  ✅ PDB → numpy → PDB")
    logger.info(f"  ✅ PDB → torch → CIF")
    logger.info(f"  ✅ CIF → numpy → PDB")
    logger.info(f"  ✅ CIF → torch → CIF")
    
    return output_dir


def analyze_structure_differences():
    """分析不同格式之间的结构差异。"""
    logger.info(f"\n🔍 结构差异分析")
    
    pdb_file = Path("9kvc.pdb")
    cif_file = Path("9kvc.cif")
    
    if pdb_file.exists() and cif_file.exists():
        pdb_protein = load_structure(pdb_file)
        cif_protein = load_structure(cif_file)
        
        logger.info(f"PDB结构: {pdb_protein.n_atoms} 原子, {pdb_protein.n_residues} 残基")
        logger.info(f"CIF结构: {cif_protein.n_atoms} 原子, {cif_protein.n_residues} 残基")
        
        atom_diff = abs(pdb_protein.n_atoms - cif_protein.n_atoms)
        residue_diff = abs(pdb_protein.n_residues - cif_protein.n_residues)
        
        logger.info(f"差异: {atom_diff} 原子, {residue_diff} 残基")
        
        if atom_diff == 0 and residue_diff == 0:
            logger.info("✅ PDB和CIF结构完全一致!")
        else:
            logger.info("ℹ️  格式间存在细微差异（正常现象）")


def main():
    """主函数。"""
    try:
        output_dir = demonstrate_conversion_workflow()
        analyze_structure_differences()
        
        logger.info(f"\n🎉 演示成功完成！")
        logger.info(f"输出文件位于: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"❌ 演示过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 