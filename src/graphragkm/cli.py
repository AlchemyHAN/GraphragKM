#!/usr/bin/env python3
"""
graphragkm - GraphRAG驱动的AI本体生成工具
命令行界面
"""
import sys
from typing import Optional

import click
from rich.console import Console

# 导入主模块
from .main import main_entry

console = Console()


@click.group(help="graphragkm - GraphRAG驱动的AI本体生成工具")
def cli():
    """AIOntology命令行工具入口点"""
    pass


@cli.command(name="run", help="初始化并运行本体生成流程")
@click.option(
    "--input-pdf",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    help="输入的PDF文件路径",
)
def init_command(input_pdf: Optional[str]):
    """初始化并运行本体生成流程"""
    # 直接调用主函数
    main_entry(input_pdf)


@cli.command(name="version", help="显示版本信息")
def version():
    """显示版本信息"""
    from . import __version__

    console.print(f"[green]graphragkm 版本: {__version__}[/]")


def main():
    """命令行入口点"""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]错误: {str(e)}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
