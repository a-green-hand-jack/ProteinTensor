# 批量转换工具使用指南

本项目提供了强大的批量转换工具，可以将PDB/CIF蛋白质结构文件转换为高效的ProteinTensor格式。

## 🚀 功能特性

- **并行处理**: 支持多进程并行转换，默认使用CPU核心数的一半
- **多种后端**: 支持numpy (.npz) 和 PyTorch (.pt) 两种存储格式  
- **目录结构**: 可选择保留原始目录结构或平铺输出
- **递归搜索**: 支持递归扫描子目录中的结构文件
- **文件格式**: 支持 .pdb, .ent, .cif, .mmcif 多种格式
- **元数据保存**: 自动保存源文件信息、原子数量等元数据
- **命令行接口**: 提供完整的CLI工具，易于集成到工作流程中

## 📋 API使用方式

### 简单转换

```python
from protein_tensor import convert_structures

# 转换单个文件
results = convert_structures(
    input_path="protein.pdb",
    output_dir="tensors/",
    backend="numpy"
)

# 批量转换文件夹（递归）
results = convert_structures(
    input_path="structures/", 
    output_dir="tensors/",
    backend="torch",
    n_workers=8,
    recursive=True
)
```

### 使用BatchConverter类

```python
from protein_tensor import BatchConverter

# 创建转换器
converter = BatchConverter(
    backend="numpy",
    n_workers=4,
    preserve_structure=True
)

# 批量转换
results = converter.convert_batch(
    input_path="data/",
    output_dir="output/",
    recursive=True
)

print(f"成功转换: {results['success']}/{results['total']}")
```

## 🛠️ 命令行工具使用

### 基本语法

```bash
python scripts/batch_convert.py [输入路径] [输出目录] [选项]
```

### 常用示例

```bash
# 转换单个文件
python scripts/batch_convert.py protein.pdb output/

# 递归转换整个目录
python scripts/batch_convert.py structures/ tensors/ --recursive

# 使用PyTorch后端，8个并行进程
python scripts/batch_convert.py data/ output/ --backend torch --workers 8

# 平铺输出，不保留目录结构
python scripts/batch_convert.py structures/ output/ --no-preserve-structure

# 详细日志输出
python scripts/batch_convert.py data/ output/ --verbose
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--backend` | `-b` | 存储后端 (numpy/torch) | numpy |
| `--workers` | `-w` | 并行进程数 | CPU核心数/2 |
| `--recursive` | `-r` | 递归搜索子目录 | False |
| `--no-preserve-structure` | - | 不保留目录结构 | False |
| `--verbose` | `-v` | 详细日志输出 | False |

## 📁 输出格式

### numpy格式 (.npz)

```python
import numpy as np

# 加载数据
data = np.load("protein.npz", allow_pickle=True)

# 访问数据
coordinates = data['coordinates']  # (N, 3) 原子坐标
atom_types = data['atom_types']    # (N,) 原子类型索引
residue_types = data['residue_types']  # (N,) 残基类型索引
chain_ids = data['chain_ids']      # (N,) 链ID索引
residue_numbers = data['residue_numbers']  # (N,) 残基编号
metadata = data['metadata'].item()  # 元数据字典
```

### PyTorch格式 (.pt)

```python
import torch

# 加载数据
data = torch.load("protein.pt")

# 访问数据
coordinates = data['coordinates']  # torch.Tensor (N, 3)
atom_types = data['atom_types']    # torch.Tensor (N,)
metadata = data['metadata']        # 元数据字典
```

### 元数据内容

```python
metadata = {
    'source_file': '/path/to/original.pdb',
    'n_atoms': 7502,
    'n_residues': 321,
    'backend': 'numpy'  # or 'torch'
}
```

## 🎯 性能优化建议

1. **并行进程数**: 对于I/O密集的任务，可以设置比CPU核心数更多的进程
2. **存储后端**: numpy格式压缩更好，torch格式加载更快
3. **目录结构**: 平铺输出可以提高某些工作流程的效率
4. **批处理大小**: 对于大量小文件，建议增加并行进程数

## 🔍 故障排除

### 常见错误

1. **内存不足**: 减少并行进程数或处理较小的批次
2. **文件权限**: 确保对输入和输出目录有适当的读写权限
3. **依赖缺失**: 确保安装了所有必需的依赖包

### 调试方法

```bash
# 使用详细模式查看详细信息
python scripts/batch_convert.py data/ output/ --verbose

# 单进程模式便于调试
python scripts/batch_convert.py data/ output/ --workers 1 --verbose
```

## 📈 示例工作流程

### 大规模数据集处理

```bash
# 1. 递归扫描并转换为numpy格式
python scripts/batch_convert.py /data/pdb_files/ /output/numpy_tensors/ \
    --backend numpy --recursive --workers 16

# 2. 转换为PyTorch格式用于模型训练  
python scripts/batch_convert.py /data/pdb_files/ /output/torch_tensors/ \
    --backend torch --recursive --workers 8 --no-preserve-structure
```

### 单项目处理

```python
from protein_tensor import convert_structures

# 转换项目数据
results = convert_structures(
    input_path="my_proteins/",
    output_dir="processed/",
    backend="numpy",
    recursive=True,
    preserve_structure=True
)

print(f"处理完成: {results['success']}/{results['total']} 个文件")
``` 