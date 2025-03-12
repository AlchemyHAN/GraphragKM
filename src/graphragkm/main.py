"""
graphragkm - 主模块
"""

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import PDFProcessor, MarkdownProcessor
from .config.config import Config
from .inference_processor import InferenceProcessor
from .owl_generator import OWLGenerator
from .uml_generator import PlantUMLGenerator

DEFAULT_OUTPUT_DIR = "output"
MD_OUTPUT_FILENAME = "output.md"
GRAPHRAG_INPUT_FILENAME = "input.txt"
CONFIG_FILENAME = "config.yaml"

console = Console()


def load_graphrag_configs(config: Config, project_dir: Path) -> dict:
    """加载并更新GraphRAG配置

    Args:
        config: 配置对象
        project_dir: 项目根目录路径

    Returns:
        更新后的GraphRAG配置字典
    """

    # 加载GraphRAG配置
    graphrag_settings_path = project_dir / "settings.yaml"

    with open(graphrag_settings_path, "r", encoding="utf-8") as f:
        graphrag_settings = yaml.safe_load(f)

    # 更新模型配置
    chat_model = graphrag_settings["models"]["default_chat_model"]
    embedding_model = graphrag_settings["models"]["default_embedding_model"]

    chat_model.update(
        {
            "api_key": config.chat_model_api_key,
            "api_base": config.chat_model_api_base,
            "model": config.chat_model_name,
            "encoding_model": chat_model.get("encoding_model", "cl100k_base"),
        }
    )

    embedding_model.update(
        {
            "api_key": config.embedding_model_api_key,
            "api_base": config.embedding_model_api_base,
            "model": config.embedding_model_name,
            "encoding_model": embedding_model.get("encoding_model", "cl100k_base"),
        }
    )

    console.print("[green]✓ GraphRAG配置加载完成[/]")
    return graphrag_settings


def check_config(config_path: Path) -> bool:
    """检查配置是否完整"""
    try:
        config = Config.from_yaml(str(config_path))
        is_valid, error_msg = config.validate()

        if not is_valid:
            console.print(f"[red]错误: {error_msg}[/]")
            return False

        return True

    except FileNotFoundError:
        console.print(f"[red]错误: 找不到配置文件 {config_path}[/]")
        return False
    except yaml.YAMLError:
        console.print(f"[red]错误: 配置文件格式不正确 {config_path}[/]")
        return False


def ensure_config() -> bool:
    """确保配置文件存在且完整"""
    config_dir = Path(__file__).parent.parent.parent
    config_path = config_dir / CONFIG_FILENAME

    if not config_path.exists():
        console.print("[yellow]未找到配置文件，正在创建模板...[/]")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        template = {
            "api": {
                "mineru_upload_url": "https://mineru.net/api/v4/file-urls/batch",
                "mineru_results_url_template": "https://mineru.net/api/v4/extract-results/batch/{}",
                "mineru_token": "YOUR_MINERU_TOKEN",
                "chat_model_api_key": "YOUR_CHAT_MODEL_API_KEY",
                "chat_model_api_base": "https://api.deepseek.com",
                "chat_model_name": "deepseek-chat",
                "embedding_model_api_key": "YOUR_EMBEDDING_MODEL_API_KEY",
                "embedding_model_api_base": "https://open.bigmodel.cn/api/paas/v4/",
                "embedding_model_name": "embedding-3",
            },
            "app": {
                "owl_namespace": "https://example.com/",
                "max_concurrent_requests": 25,
                "doc_language": "en",
            },
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, allow_unicode=True)

        console.print(
            f"""[yellow]已创建配置文件模板: {config_path}
请编辑配置文件，填入正确的配置信息后重新运行程序。[/]"""
        )
        return False

    return check_config(config_path)


async def build_graphrag_index(graphrag_config):
    """构建GraphRAG索引

    Args:
        graphrag_config: GraphRAG配置对象
    """
    import graphrag.api as api

    console.print("[blue]开始构建GraphRAG索引...[/]")
    index_result = await api.build_index(config=graphrag_config)

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
    ) as progress:
        for workflow_result in index_result:
            status = (
                f"[red]error: {workflow_result.errors}[/]"
                if workflow_result.errors
                else "[green]success[/]"
            )
            progress.add_task(
                f"[blue]Workflow: {workflow_result.workflow} - Status: {status}[/]",
                completed=True,
            )


def process_markdown_files(
        progress: Progress,
        md_processor: MarkdownProcessor,
        md_files: list[Path],
        output_dir: Path,
) -> None:
    """处理所有Markdown文件

    Args:
        progress: 进度条对象
        md_processor: Markdown处理器
        md_files: Markdown文件列表
        output_dir: 输出目录
    """
    md_task = progress.add_task(
        "[cyan]处理Markdown文件中的图片[/]", total=len(md_files)
    )

    for md_file in md_files:
        output_path = output_dir / MD_OUTPUT_FILENAME
        try:
            md_processor.process_markdown_file(str(md_file), str(output_path))
            progress.update(md_task, advance=1)
        except Exception as e:
            console.print(f"[red]错误: 处理失败 {md_file.name}: {str(e)}[/]")


def prepare_graphrag_input(output_dir: Path, graphrag_input_dir: Path) -> None:
    """准备GraphRAG输入文件

    Args:
        output_dir: 输出目录
        graphrag_input_dir: GraphRAG输入目录
    """
    if not graphrag_input_dir.exists():
        graphrag_input_dir.mkdir(parents=True, exist_ok=True)

    source_file = output_dir / MD_OUTPUT_FILENAME
    target_file = graphrag_input_dir / GRAPHRAG_INPUT_FILENAME
    shutil.copy2(str(source_file), str(target_file))


async def run_inference_pipeline(inference_processor: InferenceProcessor):
    """进行推理

    Args:
        inference_processor: 推理处理器实例
    """
    console.print("[cyan]开始推理...[/]")

    # 推理实体属性
    await inference_processor.infer_all_attributes()

    # 推理实体关系
    await inference_processor.infer_all_relationships()

    # 计算实体嵌入
    await inference_processor.compute_all_embeddings()

    # 聚类
    await inference_processor.cluster_entities()

    console.print("[green]✓ 推理执行完成[/]")


def main_entry(input_pdf: Optional[str] = None):
    """AI 本体生成工具"""
    console.print("[cyan]===== GraphragKM =====\n[/]")

    if not ensure_config():
        sys.exit(1)
    console.print("[green]✓ 配置检查通过[/]")

    # 初始化路径
    project_dir = Path(__file__).parent.parent.parent
    config_file = project_dir / CONFIG_FILENAME
    output_dir = project_dir / DEFAULT_OUTPUT_DIR
    config = Config.from_yaml(str(config_file))

    # 交互式获取输入文件
    if not input_pdf:
        input_pdf = click.prompt(
            "请输入PDF文件路径\n", type=click.Path(exists=True, dir_okay=False)
        )

    if input_pdf is not None:
        input_pdf_path = Path(input_pdf)
    else:
        console.print("[red]错误: 未提供PDF文件路径[/]")
        sys.exit(1)

    output_dir_path = Path(output_dir)
    input_dir_path = project_dir / "input"

    # 清理输出目录
    if output_dir_path.exists():
        console.print(f"[yellow]警告: 输出目录已存在，是否清空？[/]")
        response = click.prompt(
            "请输入 (y/n)", type=click.Choice(["y", "n"], case_sensitive=False)
        )
        if response.lower() == "y":
            shutil.rmtree(output_dir_path)

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
    ) as progress:
        # 处理PDF
        pdf_task = progress.add_task(
            f"[cyan]处理PDF文件: {input_pdf_path.name}[/]", total=None
        )

        pdf_processor = PDFProcessor(str(config_file))
        pdf_processor.process_pdf(str(input_pdf_path), str(output_dir_path))
        progress.update(pdf_task, completed=True)

        # 处理Markdown
        md_processor = MarkdownProcessor(str(output_dir_path))
        md_files = list(output_dir_path.glob("*.md"))

        if not md_files:
            console.print("[yellow]警告: 输出目录中没有找到Markdown文件[/]")
            return

        process_markdown_files(progress, md_processor, md_files, output_dir_path)

        # 初始化GraphRAG
        init_task = progress.add_task("[cyan]初始化 GraphRAG 项目[/]", total=None)
        from graphrag.cli.initialize import initialize_project_at

        initialize_project_at(project_dir, True)
        progress.update(init_task, completed=True)

        prepare_graphrag_input(output_dir_path, input_dir_path)

        index_task = progress.add_task(
            "[cyan]构建 GraphRAG 索引[/]", total=None
        )

        graphrag_settings = load_graphrag_configs(config, project_dir)
        from graphrag.config.create_graphrag_config import create_graphrag_config

        graphrag_config = create_graphrag_config(
            values=graphrag_settings, root_dir=str(project_dir)
        )

        progress.update(index_task, completed=True)

    # 执行GraphRAG索引构建
    asyncio.run(build_graphrag_index(graphrag_config))
    console.print("[green]✓ GraphRAG 索引构建完成！[/]")

    # 处理推理和生成
    inference_processor = InferenceProcessor(config, output_dir_path)
    asyncio.run(run_inference_pipeline(inference_processor))

    # 生成 OWL 和 UML
    console.print("[cyan]开始生成本体和UML模型...[/]")

    owl_generator = OWLGenerator(config=config, input_path=str(output_dir_path))
    owl_generator.run()

    uml_generator = PlantUMLGenerator(str(output_dir_path))
    uml_generator.run()

    console.print("\n[green]✓ 所有步骤处理完成！[/]")
    console.print(f"[blue]处理结果保存在: {output_dir_path.absolute()}[/]")
    console.print("[cyan]===== 处理结束 =====\n[/]")


if __name__ == "__main__":
    main_entry()
