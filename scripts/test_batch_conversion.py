#!/usr/bin/env python3
"""
测试批量转换功能的演示脚本
"""
import logging
import sys
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_tensor import convert_structures, BatchConverter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_structure():
    """创建测试目录结构。"""
    test_dir = Path("test_structures")
    test_dir.mkdir(exist_ok=True)
    
    # 创建子目录结构
    (test_dir / "proteins").mkdir(exist_ok=True)
    (test_dir / "ligands").mkdir(exist_ok=True)
    
    # 复制测试文件到不同位置
    import shutil
    
    # 根目录的文件
    if Path("9kvc.pdb").exists():
        shutil.copy("9kvc.pdb", test_dir / "protein_1.pdb")
    if Path("9kvc.cif").exists():
        shutil.copy("9kvc.cif", test_dir / "protein_1.cif")
    
    # 子目录的文件
    if Path("9kvc.pdb").exists():
        shutil.copy("9kvc.pdb", test_dir / "proteins" / "complex.pdb")
    if Path("9kvc.cif").exists():
        shutil.copy("9kvc.cif", test_dir / "ligands" / "small_molecule.cif")
    
    logger.info(f"创建测试目录结构: {test_dir}")
    return test_dir


def test_single_file_conversion():
    """测试单文件转换。"""
    logger.info("🧪 测试 1: 单文件转换")
    
    pdb_file = Path("9kvc.pdb")
    if not pdb_file.exists():
        logger.warning("PDB文件不存在，跳过单文件测试")
        return
    
    output_dir = Path("single_conversion_output")
    
    start_time = time.time()
    results = convert_structures(
        input_path=pdb_file,
        output_dir=output_dir,
        backend="numpy",
        n_workers=1
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"单文件转换结果: {results}")
    logger.info(f"转换耗时: {elapsed_time:.2f} 秒")
    
    # 验证输出文件
    output_files = list(output_dir.glob("*.npz"))
    logger.info(f"输出文件: {[f.name for f in output_files]}")


def test_batch_conversion_numpy():
    """测试批量转换 (numpy后端)。"""
    logger.info("🧪 测试 2: 批量转换 (numpy后端)")
    
    test_dir = create_test_structure()
    output_dir = Path("batch_numpy_output")
    
    start_time = time.time()
    results = convert_structures(
        input_path=test_dir,
        output_dir=output_dir,
        backend="numpy",
        recursive=True,
        preserve_structure=True,
        n_workers=2
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"numpy批量转换结果: {results}")
    logger.info(f"转换耗时: {elapsed_time:.2f} 秒")
    
    # 显示输出结构
    logger.info("输出文件结构:")
    for file in sorted(output_dir.rglob("*")):
        if file.is_file():
            logger.info(f"  {file.relative_to(output_dir)}")


def test_batch_conversion_torch():
    """测试批量转换 (torch后端)。"""
    logger.info("🧪 测试 3: 批量转换 (torch后端)")
    
    test_dir = Path("test_structures")
    if not test_dir.exists():
        logger.warning("测试目录不存在，跳过torch批量测试")
        return
    
    output_dir = Path("batch_torch_output")
    
    start_time = time.time()
    results = convert_structures(
        input_path=test_dir,
        output_dir=output_dir,
        backend="torch",
        recursive=True,
        preserve_structure=False,  # 平铺输出
        n_workers=1
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"torch批量转换结果: {results}")
    logger.info(f"转换耗时: {elapsed_time:.2f} 秒")
    
    # 显示输出文件
    output_files = list(output_dir.glob("*.pt"))
    logger.info(f"输出文件: {[f.name for f in output_files]}")


def test_load_converted_data():
    """测试加载转换后的数据。"""
    logger.info("🧪 测试 4: 加载转换后的数据")
    
    # 测试加载numpy数据
    numpy_dir = Path("batch_numpy_output")
    if numpy_dir.exists():
        numpy_files = list(numpy_dir.rglob("*.npz"))
        if numpy_files:
            import numpy as np
            data = np.load(numpy_files[0], allow_pickle=True)
            logger.info(f"numpy数据文件: {numpy_files[0]}")
            logger.info(f"  包含的键: {list(data.keys())}")
            logger.info(f"  坐标形状: {data['coordinates'].shape}")
            logger.info(f"  元数据: {data['metadata'].item()}")
    
    # 测试加载torch数据
    torch_dir = Path("batch_torch_output")
    if torch_dir.exists():
        torch_files = list(torch_dir.glob("*.pt"))
        if torch_files:
            try:
                import torch
                data = torch.load(torch_files[0])
                logger.info(f"torch数据文件: {torch_files[0]}")
                logger.info(f"  包含的键: {list(data.keys())}")
                logger.info(f"  坐标形状: {data['coordinates'].shape}")
                logger.info(f"  元数据: {data['metadata']}")
            except ImportError:
                logger.warning("PyTorch未安装，跳过torch数据加载测试")


def test_converter_class():
    """测试BatchConverter类的直接使用。"""
    logger.info("🧪 测试 5: BatchConverter类直接使用")
    
    # 创建转换器实例
    converter = BatchConverter(
        backend="numpy",
        n_workers=1,
        preserve_structure=True
    )
    
    # 测试单文件转换
    pdb_file = Path("9kvc.pdb")
    if pdb_file.exists():
        output_file = Path("class_test_output") / "test.npz"
        success = converter.convert_single(pdb_file, output_file)
        logger.info(f"BatchConverter单文件转换: {'成功' if success else '失败'}")
        
        if success and output_file.exists():
            import numpy as np
            data = np.load(output_file, allow_pickle=True)
            logger.info(f"  文件大小: {output_file.stat().st_size // 1024} KB")
            logger.info(f"  原子数量: {data['metadata'].item()['n_atoms']}")


def cleanup_test_files():
    """清理测试文件。"""
    logger.info("🧹 清理测试文件")
    
    import shutil
    cleanup_dirs = [
        "test_structures",
        "single_conversion_output", 
        "batch_numpy_output",
        "batch_torch_output",
        "class_test_output"
    ]
    
    for dir_name in cleanup_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"删除目录: {dir_path}")


def main():
    """主函数，运行所有测试。"""
    logger.info("🚀 开始批量转换功能测试")
    
    try:
        # 运行各项测试
        test_single_file_conversion()
        print()
        
        test_batch_conversion_numpy()
        print()
        
        test_batch_conversion_torch()
        print()
        
        test_load_converted_data()
        print()
        
        test_converter_class()
        print()
        
        logger.info("✅ 所有测试完成!")
        
        # 询问是否清理测试文件
        response = input("\n是否清理测试文件? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_test_files()
        else:
            logger.info("保留测试文件供查看")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 