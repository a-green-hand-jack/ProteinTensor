# Protein Tensor

一个强大的Python库，用于在PDB/CIF蛋白质结构格式和numpy/PyTorch张量之间进行高效转换。

## 🚀 功能特点

- **多格式支持**: 支持PDB和CIF格式的蛋白质结构文件
- **双向转换**: 结构文件 ↔ numpy数组 ↔ PyTorch张量
- **特征提取**: 原子坐标、原子类型、残基类型、链ID等信息
- **批量处理**: 支持大规模数据集的并行转换
- **数据完整性**: 确保转换过程中数据的准确性和一致性
- **易于集成**: 简洁的API设计，便于集成到机器学习流水线

## 📦 安装

### 方法1: 从源码安装（推荐用于开发）

```bash
# 克隆仓库
git clone git@github.com:a-green-hand-jack/ProteinTensor.git
cd ProteinTensor

# 使用uv创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate.bat  # Windows

# 安装开发版本
uv pip install -e ".[dev]"
```

### 方法2: 在其他项目中使用

如果您想在其他项目中使用这个库：

```bash
# 在您的项目目录中创建虚拟环境
cd your-project/
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 从本地路径安装（假设protein-tensor在相对路径）
uv pip install -e /path/to/ProteinTensor

# 或者直接从Git仓库安装
uv pip install git+git@github.com:a-green-hand-jack/ProteinTensor.git
```

### 方法3: 仅安装运行时依赖

```bash
# 只安装核心依赖，不包含开发工具
uv pip install -e .
```

## 💡 核心概念

### ProteinTensor类

`ProteinTensor`是核心数据容器，包含：

- **coordinates**: 原子坐标 `(n_atoms, 3)`
- **atom_types**: 原子类型索引 `(n_atoms,)`
- **residue_types**: 残基类型索引 `(n_atoms,)`
- **chain_ids**: 链ID索引 `(n_atoms,)`
- **residue_numbers**: 残基编号 `(n_atoms,)`

## 🔧 基础使用

### 单文件转换

```python
import protein_tensor as pt
import numpy as np
import torch

# 1. 加载蛋白质结构
protein = pt.load_structure("example.pdb")
print(f"加载了 {protein.n_atoms} 个原子，{protein.n_residues} 个残基")

# 2. 转换为numpy格式
numpy_data = protein.to_numpy()
print(f"坐标形状: {numpy_data['coordinates'].shape}")
print(f"原子类型: {numpy_data['atom_types'].shape}")

# 3. 转换为PyTorch格式
torch_data = protein.to_torch()
coords_tensor = torch_data['coordinates']  # torch.Tensor
print(f"PyTorch张量设备: {coords_tensor.device}")

# 4. 数据操作
protein.center_coordinates()  # 居中坐标
distances = pt.calculate_distances(coords_tensor)  # 计算距离矩阵

# 5. 保存结构
pt.save_structure(protein, "output.pdb")
pt.save_structure(protein, "output.cif")
```

### 从张量重建结构

```python
# 从numpy数组创建ProteinTensor
protein_from_numpy = pt.ProteinTensor(
    coordinates=numpy_data['coordinates'],
    atom_types=numpy_data['atom_types'],
    residue_types=numpy_data['residue_types'],
    chain_ids=numpy_data['chain_ids'],
    residue_numbers=numpy_data['residue_numbers']
)

# 保存重建的结构
pt.save_structure(protein_from_numpy, "reconstructed.pdb")

# 从PyTorch张量创建
protein_from_torch = pt.ProteinTensor(
    coordinates=torch_data['coordinates'],
    atom_types=torch_data['atom_types'],
    # ... 其他属性
)
```

## 🚀 批量转换工具

### Python API

```python
from protein_tensor import convert_structures

# 转换单个文件
results = convert_structures(
    input_path="protein.pdb",
    output_dir="./output",
    backend="numpy",  # 或 "torch"
    n_workers=4
)

# 批量转换文件夹
results = convert_structures(
    input_path="./pdb_files/",
    output_dir="./tensor_output/",
    backend="torch",
    recursive=True,  # 递归扫描子目录
    n_workers=8,     # 并行进程数
    preserve_structure=True  # 保持目录结构
)

print(f"成功转换: {results['successful']}")
print(f"失败文件: {results['failed_files']}")
```

### 命令行工具

```bash
# 转换单个文件
python scripts/batch_convert.py protein.pdb -o ./output --backend numpy

# 批量转换（numpy格式）
python scripts/batch_convert.py ./pdb_files/ -o ./numpy_output --backend numpy --workers 8

# 批量转换（PyTorch格式）
python scripts/batch_convert.py ./structures/ -o ./torch_output --backend torch --recursive

# 查看帮助
python scripts/batch_convert.py --help
```

### 输出格式

转换后的文件包含完整的结构信息：

**Numpy格式 (.npz)**:
```python
data = np.load("protein.npz", allow_pickle=True)
coordinates = data['coordinates']    # (n_atoms, 3)
atom_types = data['atom_types']      # (n_atoms,)
metadata = data['metadata'].item()  # dict with original info
```

**PyTorch格式 (.pt)**:
```python
data = torch.load("protein.pt")
coordinates = data['coordinates']    # torch.Tensor (n_atoms, 3)
metadata = data['metadata']         # dict with original info
```

## 🔬 数据加载和验证

### 加载转换后的数据

```python
import numpy as np
import torch

# 加载numpy格式
numpy_data = np.load("protein.npz", allow_pickle=True)
protein_numpy = pt.ProteinTensor(
    coordinates=numpy_data['coordinates'],
    atom_types=numpy_data['atom_types'],
    residue_types=numpy_data['residue_types'],
    chain_ids=numpy_data['chain_ids'],
    residue_numbers=numpy_data['residue_numbers']
)

# 加载PyTorch格式
torch_data = torch.load("protein.pt")
protein_torch = pt.ProteinTensor(**torch_data)

# 验证数据一致性
original = pt.load_structure("original.pdb")
np.testing.assert_allclose(
    original.to_numpy()['coordinates'],
    protein_numpy.coordinates,
    rtol=1e-6
)
```

### 交叉验证

```python
# 完整的往返转换测试
original = pt.load_structure("protein.pdb")

# PDB → numpy → 新PDB → 验证
numpy_data = original.to_numpy()
reconstructed = pt.ProteinTensor(**numpy_data)
pt.save_structure(reconstructed, "test.pdb")
reloaded = pt.load_structure("test.pdb")

# 验证坐标一致性
assert np.allclose(
    original.to_numpy()['coordinates'],
    reloaded.to_numpy()['coordinates'],
    rtol=1e-3
)
```

## 🔧 二次开发指南

### 开发环境设置

```bash
# 克隆并设置开发环境
git clone git@github.com:a-green-hand-jack/ProteinTensor.git
cd ProteinTensor
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行测试
uv run pytest tests/ -v

# 代码质量检查
uv run black src/ tests/           # 格式化
uv run flake8 src/ tests/         # 风格检查
uv run mypy src/                  # 类型检查
```

### 扩展功能

```python
from protein_tensor import ProteinTensor
import numpy as np

class EnhancedProteinTensor(ProteinTensor):
    """扩展的蛋白质张量类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_features = None
    
    def compute_custom_features(self):
        """计算自定义特征"""
        if self._cached_features is None:
            # 实现您的特征计算逻辑
            coords = self.to_numpy()['coordinates']
            # ... 自定义计算
            self._cached_features = features
        return self._cached_features
    
    def save_with_features(self, filepath):
        """保存包含特征的数据"""
        data = self.to_numpy()
        data['custom_features'] = self.compute_custom_features()
        np.savez_compressed(filepath, **data)

# 使用扩展类
enhanced_protein = EnhancedProteinTensor.from_file("protein.pdb")
enhanced_protein.save_with_features("enhanced_protein.npz")
```

### 集成到机器学习流水线

```python
import torch
from torch.utils.data import Dataset, DataLoader
from protein_tensor import load_structure

class ProteinDataset(Dataset):
    """蛋白质数据集类"""
    
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        protein = load_structure(self.file_list[idx])
        data = protein.to_torch()
        
        if self.transform:
            data = self.transform(data)
            
        return data

# 使用示例
dataset = ProteinDataset(['protein1.pdb', 'protein2.pdb'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    coordinates = batch['coordinates']  # (batch_size, n_atoms, 3)
    # 进行模型训练...
```

## 📊 性能优化

### 批量处理最佳实践

```python
# 对于大型数据集，使用批量转换
results = convert_structures(
    input_path="large_dataset/",
    output_dir="processed/",
    backend="torch",
    n_workers=min(16, os.cpu_count()),  # 适当的并行数
    recursive=True
)

# 监控进度
from tqdm import tqdm
for file_path in tqdm(file_list):
    protein = load_structure(file_path)
    # 处理...
```

## 🧪 测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 生成覆盖率报告
uv run pytest tests/ --cov=src --cov-report=html

# 运行特定测试
uv run pytest tests/test_io.py::TestIO::test_complete_conversion_workflow -v
```

## 📁 项目结构

```
protein-tensor/
├── src/protein_tensor/          # 核心库代码
│   ├── __init__.py             # 包初始化和导出
│   ├── core.py                 # ProteinTensor核心类
│   ├── io.py                   # 文件输入输出操作
│   ├── utils.py                # 工具函数和常量
│   └── batch_converter.py      # 批量转换工具
├── tests/                      # 测试代码
│   ├── test_io.py             # I/O操作测试
│   └── test_batch_conversion.py # 批量转换测试
├── scripts/                    # 脚本工具
│   ├── batch_convert.py        # 批量转换CLI工具
│   └── conversion_demo.py      # 转换演示脚本
├── docs/                       # 文档
│   ├── DEVELOPMENT.md          # 开发指南
│   ├── CHANGELOG.md            # 更新日志
│   └── BATCH_CONVERSION.md     # 批量转换详细说明
├── pyproject.toml              # 项目配置
├── README.md                   # 本文档
└── LICENSE                     # 许可证
```

## 📚 依赖项

### 核心依赖
- **numpy** (≥1.21.0): 数值计算基础
- **torch** (≥1.9.0): PyTorch张量操作
- **biopython** (≥1.79): 蛋白质结构文件解析
- **pandas** (≥1.3.0): 数据处理和分析
- **typing-extensions** (≥4.0.0): 类型注解支持

### 开发依赖
- **pytest** + **pytest-cov**: 测试框架和覆盖率
- **black**: 代码格式化
- **flake8**: 代码风格检查  
- **mypy**: 静态类型检查
- **isort**: 导入排序

## 🐛 故障排除

### 常见问题

1. **内存不足**：处理大型蛋白质时减少并行进程数
2. **精度问题**：使用适当的数值容差进行比较
3. **文件格式**：确保PDB/CIF文件格式正确

### 获取帮助

- 查看测试用例了解使用模式
- 检查日志输出获取详细错误信息
- 参考`docs/`目录中的详细文档

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和Pull Request！请参考 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) 了解开发指南。 