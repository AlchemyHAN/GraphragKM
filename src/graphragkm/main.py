"""
graphragkm - Main module
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
from .knowledge_enricher import KnowledgeEnricher

DEFAULT_OUTPUT_DIR = "output"
MD_OUTPUT_FILENAME = "output.md"
GRAPHRAG_INPUT_FILENAME = "input.txt"
CONFIG_FILENAME = "config.yaml"

console = Console()


def load_graphrag_configs(config: Config, project_dir: Path) -> dict:
    """Load and update GraphRAG configuration

    Args:
        config: Configuration object
        project_dir: Project root directory path

    Returns:
        Updated GraphRAG configuration dictionary
    """

    # Load GraphRAG configuration
    graphrag_settings_path = project_dir / "settings.yaml"

    with open(graphrag_settings_path, "r", encoding="utf-8") as f:
        graphrag_settings = yaml.safe_load(f)

    # Update model configuration
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

    console.print("[green]✓ GraphRAG configuration loaded[/]")
    return graphrag_settings


def check_config(config_path: Path) -> bool:
    """Check if configuration is complete"""
    try:
        config = Config.from_yaml(str(config_path))
        is_valid, error_msg = config.validate()

        if not is_valid:
            console.print(f"[red]Error: {error_msg}[/]")
            return False

        return True

    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file not found {config_path}[/]")
        return False
    except yaml.YAMLError:
        console.print(f"[red]Error: Invalid configuration file format {config_path}[/]")
        return False


def ensure_config() -> bool:
    """Ensure configuration file exists and is complete"""
    config_dir = Path(__file__).parent.parent.parent
    config_path = config_dir / CONFIG_FILENAME

    if not config_path.exists():
        console.print("[yellow]Configuration file not found, creating template...[/]")
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
            f"""[yellow]Configuration template created: {config_path}
Please edit the configuration file, fill in the correct information and run the program again.[/]"""
        )
        return False

    return check_config(config_path)


async def build_graphrag_index(graphrag_config):
    """Build GraphRAG index

    Args:
        graphrag_config: GraphRAG configuration object
    """
    import graphrag.api as api

    console.print("[blue]Starting to build GraphRAG index...[/]")
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
    """Process all Markdown files

    Args:
        progress: Progress bar object
        md_processor: Markdown processor
        md_files: List of Markdown files
        output_dir: Output directory
    """
    md_task = progress.add_task(
        "[cyan]Processing images in Markdown files[/]", total=len(md_files)
    )

    for md_file in md_files:
        output_path = output_dir / MD_OUTPUT_FILENAME
        try:
            md_processor.process_markdown_file(str(md_file), str(output_path))
            progress.update(md_task, advance=1)
        except Exception as e:
            console.print(f"[red]Error: Processing failed {md_file.name}: {str(e)}[/]")


def prepare_graphrag_input(output_dir: Path, graphrag_input_dir: Path) -> None:
    """Prepare GraphRAG input files

    Args:
        output_dir: Output directory
        graphrag_input_dir: GraphRAG input directory
    """
    if not graphrag_input_dir.exists():
        graphrag_input_dir.mkdir(parents=True, exist_ok=True)

    source_file = output_dir / MD_OUTPUT_FILENAME
    target_file = graphrag_input_dir / GRAPHRAG_INPUT_FILENAME
    shutil.copy2(str(source_file), str(target_file))


async def run_inference_pipeline(inference_processor: InferenceProcessor):
    """Run inference

    Args:
        inference_processor: Inference processor instance
    """
    console.print("[cyan]Starting inference...[/]")

    # Infer entity attributes
    await inference_processor.infer_all_attributes()

    # Infer entity relationships
    await inference_processor.infer_all_relationships()

    # Compute entity embeddings
    await inference_processor.compute_all_embeddings()

    # Clustering
    await inference_processor.cluster_entities()

    console.print("[green]✓ Inference completed[/]")


def main_entry(input_pdf: Optional[str] = None):
    """AI Ontology Generation Tool"""
    console.print("[cyan]===== GraphragKM =====\n[/]")

    if not ensure_config():
        sys.exit(1)
    console.print("[green]✓ Configuration check passed[/]")

    # Initialize paths
    project_dir = Path(__file__).parent.parent.parent
    config_file = project_dir / CONFIG_FILENAME
    output_dir = project_dir / DEFAULT_OUTPUT_DIR
    config = Config.from_yaml(str(config_file))

    # Interactive input file acquisition
    if not input_pdf:
        input_pdf = click.prompt(
            "Please enter PDF file path\n", type=click.Path(exists=True, dir_okay=False)
        )

    if input_pdf is not None:
        input_pdf_path = Path(input_pdf)
    else:
        console.print("[red]Error: No PDF file path provided[/]")
        sys.exit(1)

    output_dir_path = Path(output_dir)
    input_dir_path = project_dir / "input"

    # Clean output directory
    if output_dir_path.exists():
        console.print(
            f"[yellow]Warning: Output directory exists, do you want to clear?[/]"
        )
        response = click.prompt(
            "Please enter (y/n)", type=click.Choice(["y", "n"], case_sensitive=False)
        )
        if response.lower() == "y":
            shutil.rmtree(output_dir_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Process PDF
        pdf_task = progress.add_task(
            f"[cyan]Processing PDF file: {input_pdf_path.name}[/]", total=None
        )

        pdf_processor = PDFProcessor(str(config_file))
        pdf_processor.process_pdf(str(input_pdf_path), str(output_dir_path))
        progress.update(pdf_task, completed=True)

        # Process Markdown
        md_processor = MarkdownProcessor(str(output_dir_path))
        md_files = list(output_dir_path.glob("*.md"))

        if not md_files:
            console.print(
                "[yellow]Warning: No Markdown files found in output directory[/]"
            )
            return

        process_markdown_files(progress, md_processor, md_files, output_dir_path)

        # Initialize GraphRAG
        init_task = progress.add_task(
            "[cyan]Initializing GraphRAG project[/]", total=None
        )
        from graphrag.cli.initialize import initialize_project_at

        initialize_project_at(project_dir, True)
        progress.update(init_task, completed=True)

        prepare_graphrag_input(output_dir_path, input_dir_path)

        index_task = progress.add_task("[cyan]Building GraphRAG index[/]", total=None)

        graphrag_settings = load_graphrag_configs(config, project_dir)
        from graphrag.config.create_graphrag_config import create_graphrag_config

        graphrag_config = create_graphrag_config(
            values=graphrag_settings, root_dir=str(project_dir)
        )

        progress.update(index_task, completed=True)

    # Execute GraphRAG index building
    asyncio.run(build_graphrag_index(graphrag_config))
    console.print("[green]✓ GraphRAG index building completed![/]")

    # Process inference and generation
    inference_processor = InferenceProcessor(config, output_dir_path)
    asyncio.run(run_inference_pipeline(inference_processor))

    # Generate OWL and UML
    console.print("[cyan]Starting to generate ontology and UML model...[/]")

    owl_generator = OWLGenerator(config=config, input_path=str(output_dir_path))
    owl_generator.run()

    # Knowledge Enrichment
    console.print("[cyan]Starting knowledge enrichment...[/]")
    knowledge_enricher = KnowledgeEnricher(
        config=config,
        input_path=str(output_dir_path / "ontology.owl"),
        output_path=str(output_dir_path / "enriched_ontology.owl"),
    )
    asyncio.run(knowledge_enricher.run())

    uml_generator = PlantUMLGenerator(str(output_dir_path))
    uml_generator.run()

    console.print("\n[green]✓ All steps processed![/]")
    console.print(f"[blue]Processing results saved in: {output_dir_path.absolute()}[/]")
    console.print("[cyan]===== Processing ended =====\n[/]")


if __name__ == "__main__":
    main_entry()
