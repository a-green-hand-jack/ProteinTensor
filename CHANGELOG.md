# Changelog

所有对此项目的重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

### Added
- 开发者文档 (DEVELOPMENT.md)
- 变更日志文件

## [0.1.0] - 2024-01-XX

### Added
- 初始版本发布
- 核心 `ProteinTensor` 类用于处理蛋白质结构数据
- PDB 和 CIF 文件加载功能
- numpy 数组和 PyTorch 张量转换支持
- 原子和残基特征提取工具
- 完整的测试套件，测试覆盖率 65%
- 项目文档和示例脚本
- 使用 uv 和 hatchling 的现代 Python 包构建系统

### Features
- 支持从 PDB/CIF 文件加载蛋白质结构
- 将蛋白质结构数据转换为 numpy 数组或 PyTorch 张量
- 原子坐标居中功能
- 原子类型和残基类型的索引映射
- 生化特征提取（极性、电荷、疏水性等）
- 命令行演示脚本

### Technical
- Python 3.8+ 支持
- 使用 BioPython 进行结构解析
- 类型提示支持
- 代码格式化和质量检查工具集成
- pytest 测试框架
- 代码覆盖率报告 