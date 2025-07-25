#!/usr/bin/env python3
"""
演示脚本，展示protein_tensor库的基本功能。
"""
import logging
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_tensor import load_structure, save_structure
from protein_tensor.utils import get_atom_features, get_residue_features

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """主演示函数。"""
    logger.info("开始Protein Tensor库演示")
    
    # 测试文件路径
    pdb_file = Path("9kvc.pdb")
    cif_file = Path("9kvc.cif")
    
    # 测试PDB文件加载
    if pdb_file.exists():
        logger.info(f"加载PDB文件: {pdb_file}")
        try:
            protein = load_structure(pdb_file)
            logger.info(f"成功加载PDB文件！")
            logger.info(f"  - 原子数量: {protein.n_atoms}")
            logger.info(f"  - 残基数量: {protein.n_residues}")
            logger.info(f"  - 坐标形状: {protein.coordinates.shape}")
            
            # 转换为numpy数组
            numpy_data = protein.to_numpy()
            logger.info(f"转换为numpy数组: {len(numpy_data)} 个数据项")
            
            # 如果有PyTorch，转换为张量
            try:
                torch_data = protein.to_torch()
                logger.info(f"转换为PyTorch张量: {len(torch_data)} 个数据项")
            except ImportError:
                logger.warning("PyTorch未安装，跳过张量转换")
            
            # 居中坐标
            protein.center_coordinates()
            logger.info("坐标已居中")
            
            # 保存到临时文件
            output_file = Path("output_test.pdb")
            save_structure(protein, output_file)
            logger.info(f"结构已保存到: {output_file}")
            
            # 清理临时文件
            if output_file.exists():
                output_file.unlink()
                logger.info("临时文件已清理")
                
        except Exception as e:
            logger.error(f"处理PDB文件时出错: {e}")
    else:
        logger.warning(f"PDB文件 {pdb_file} 不存在")
    
    # 测试CIF文件加载
    if cif_file.exists():
        logger.info(f"加载CIF文件: {cif_file}")
        try:
            protein = load_structure(cif_file)
            logger.info(f"成功加载CIF文件！")
            logger.info(f"  - 原子数量: {protein.n_atoms}")
            logger.info(f"  - 残基数量: {protein.n_residues}")
            
        except Exception as e:
            logger.error(f"处理CIF文件时出错: {e}")
    else:
        logger.warning(f"CIF文件 {cif_file} 不存在")
    
    # 演示工具函数
    logger.info("演示工具函数:")
    
    # 原子特征
    ca_features = get_atom_features('CA')
    logger.info(f"CA原子特征: {ca_features}")
    
    # 残基特征
    ala_features = get_residue_features('ALA')
    logger.info(f"ALA残基特征: {ala_features}")
    
    logger.info("演示完成！")


if __name__ == "__main__":
    main() 