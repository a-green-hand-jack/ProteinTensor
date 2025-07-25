# 开发者文档

## 开发环境设置

### 环境要求
- Python 3.8+
- uv (用于包管理)

### 安装开发环境

```bash
# 克隆仓库
git clone <repository-url>
cd protein-tensor

# 创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装开发依赖
uv pip install -e ".[dev]"
```

## 开发工作流

### 分支策略
- `main` 分支：稳定的发布版本
- `dev` 分支：开发版本，所有新功能首先在此分支开发

### 代码规范

1. **代码格式化**
```bash
uv run black src/ tests/
uv run isort src/ tests/
```

2. **类型检查**
```bash
uv run mypy src/
```

3. **代码风格检查**
```bash
uv run flake8 src/ tests/
```

### 测试

运行所有测试：
```bash
uv run pytest tests/ -v
```

运行特定测试：
```bash
uv run pytest tests/test_io.py::TestIO::test_load_pdb_structure -v
```

查看测试覆盖率：
```bash
uv run pytest tests/ --cov=src/protein_tensor --cov-report=html
```

### 构建包

```bash
# 构建源码包和wheel包
uv build

# 查看构建产物
ls dist/
```

## 代码架构

### 核心模块

1. **core.py** - `ProteinTensor`类，主要的数据容器
2. **io.py** - 文件输入输出操作
3. **utils.py** - 工具函数和常量定义

### 扩展指南

#### 添加新的文件格式支持

1. 在 `io.py` 中的 `_infer_format()` 函数添加新格式识别
2. 在 `load_structure()` 和 `save_structure()` 函数添加解析逻辑
3. 添加相应的测试

#### 添加新的特征提取功能

1. 在 `utils.py` 中添加新的特征提取函数
2. 在 `__init__.py` 中导出新函数
3. 添加单元测试和文档

## 提交规范

使用[Conventional Commits](https://www.conventionalcommits.org/)格式：

- `feat:` 新功能
- `fix:` 修复bug  
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建或辅助工具变动

示例：
```
feat: add support for mmtf file format
fix: handle empty protein structures gracefully
docs: update API documentation for ProteinTensor class
```

## 发布流程

1. 在 `dev` 分支完成开发和测试
2. 更新版本号（在 `src/protein_tensor/__init__.py` 和 `pyproject.toml`）
3. 更新 CHANGELOG.md
4. 合并到 `main` 分支
5. 创建标签并发布 