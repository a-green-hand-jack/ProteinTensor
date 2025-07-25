#!/usr/bin/env python3
"""
批量转换工具 - 将PDB/CIF文件转换为ProteinTensor格式

支持单文件、文件夹批量处理，并行处理，多种存储后端。
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import multiprocessing as mp

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_tensor import convert_structures, BatchConverter


def setup_logging(verbose: bool = False) -> None:
    """设置日志配置。"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format_str)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="批量转换PDB/CIF文件为ProteinTensor格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换单个文件
  python batch_convert.py input.pdb output_dir/

  # 转换文件夹 (递归)
  python batch_convert.py structures/ tensors/ --recursive

  # 使用PyTorch后端，8个并行进程
  python batch_convert.py structures/ tensors/ --backend torch --workers 8

  # 不保留目录结构，平铺输出
  python batch_convert.py structures/ tensors/ --no-preserve-structure

  # 详细日志输出
  python batch_convert.py structures/ tensors/ --verbose
        """
    )
    
    parser.add_argument(
        "input", 
        type=str,
        help="输入文件或目录路径"
    )
    
    parser.add_argument(
        "output", 
        type=str,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--backend", "-b",
        choices=["numpy", "torch"],
        default="numpy",
        help="存储后端 (default: numpy)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help=f"并行工作进程数 (default: {mp.cpu_count() // 2})"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归搜索子目录中的结构文件"
    )
    
    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="不保留目录结构，平铺输出所有文件"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志信息"
    )
    
    return parser.parse_args()


def print_summary(results: dict, elapsed_time: float) -> None:
    """打印转换结果摘要。"""
    print("\n" + "="*60)
    print("🧬 批量转换完成!")
    print("="*60)
    print(f"📊 统计信息:")
    print(f"  总文件数:     {results['total']}")
    print(f"  成功转换:     {results['success']} ✅")
    print(f"  转换失败:     {results['failed']} ❌")
    print(f"  成功率:       {results['success']/results['total']*100:.1f}%")
    print(f"⏱️  总耗时:        {elapsed_time:.2f} 秒")
    
    if results['success'] > 0:
        print(f"⚡ 平均速度:      {results['success']/elapsed_time:.2f} 文件/秒")
    
    if results['failed'] > 0:
        print(f"\n❌ 失败的文件:")
        for failed_file in results['failed_files']:
            print(f"  - {failed_file}")
    
    print("="*60)


def main() -> int:
    """主函数。"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 输入路径不存在: {input_path}")
        return 1
    
    output_path = Path(args.output)
    
    # 显示配置信息
    print("🧬 蛋白质结构批量转换工具")
    print("-" * 40)
    print(f"📂 输入路径:      {input_path}")
    print(f"📁 输出目录:      {output_path}")
    print(f"💾 存储后端:      {args.backend}")
    print(f"👥 并行进程:      {args.workers or mp.cpu_count() // 2}")
    print(f"🔍 递归搜索:      {'是' if args.recursive else '否'}")
    print(f"📋 保留结构:      {'否' if args.no_preserve_structure else '是'}")
    print("-" * 40)
    
    try:
        # 开始转换
        print("🚀 开始批量转换...")
        start_time = time.time()
        
        results = convert_structures(
            input_path=input_path,
            output_dir=output_path,
            backend=args.backend,
            n_workers=args.workers,
            recursive=args.recursive,
            preserve_structure=not args.no_preserve_structure
        )
        
        elapsed_time = time.time() - start_time
        
        # 打印结果摘要
        print_summary(results, elapsed_time)
        
        # 返回适当的退出码
        return 0 if results['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断转换过程")
        return 130
    except Exception as e:
        logger.error(f"转换过程中出现错误: {e}")
        print(f"❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 