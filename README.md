# Protein Tensor

一个用于在PDB/CIF格式和numpy/torch张量之间转换蛋白质结构数据的Python库。

## 功能特点

- 从PDB和CIF文件加载蛋白质结构数据
- 将蛋白质结构转换为numpy数组或PyTorch张量
- 将张量数据转换回PDB/CIF格式
- 支持原子坐标、原子类型、残基类型等特征提取
- 提供常用的蛋白质结构分析工具

## 安装

使用uv进行环境管理和安装：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装包及其依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"
```

## 快速开始

```python
import protein_tensor as pt

# 加载PDB文件
protein = pt.load_structure("protein.pdb")

# 转换为numpy数组
numpy_data = protein.to_numpy()
print(f"坐标形状: {numpy_data['coordinates'].shape}")

# 转换为PyTorch张量
torch_data = protein.to_torch()

# 居中坐标
protein.center_coordinates()

# 保存回PDB文件
pt.save_structure(protein, "output.pdb")
```

## 核心类和方法

### ProteinTensor类

主要的数据容器类，包含：
- `coordinates`: 原子坐标 (n_atoms, 3)
- `atom_types`: 原子类型索引 (n_atoms,)
- `residue_types`: 残基类型索引 (n_residues,)
- `chain_ids`: 链ID索引 (n_atoms,)
- `residue_numbers`: 残基编号 (n_atoms,)

### 主要方法

- `load_structure(filepath)`: 从PDB/CIF文件加载结构
- `save_structure(protein_tensor, filepath)`: 保存结构到文件
- `protein.to_numpy()`: 转换为numpy数组
- `protein.to_torch()`: 转换为PyTorch张量
- `protein.center_coordinates()`: 居中坐标

## 开发

```bash
# 运行测试
uv run pytest

# 代码格式化
uv run black src/ tests/

# 类型检查
uv run mypy src/

# 代码风格检查
uv run flake8 src/ tests/
```

## 项目结构

```
protein-tensor/
├── src/protein_tensor/     # 主要源代码
│   ├── __init__.py        # 包初始化
│   ├── core.py            # 核心ProteinTensor类
│   ├── io.py              # 输入输出操作
│   └── utils.py           # 工具函数
├── tests/                 # 测试代码
├── docs/                  # 文档
├── scripts/              # 脚本工具
├── pyproject.toml        # 项目配置
└── README.md             # 说明文档
```

## 依赖

- numpy: 数值计算
- torch: PyTorch张量操作
- biopython: 蛋白质结构解析
- pandas: 数据处理

## 许可证

MIT License 