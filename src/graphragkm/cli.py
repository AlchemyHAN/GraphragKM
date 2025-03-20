#!/usr/bin/env python3
"""
graphragkm - GraphRAG-driven AI Ontology Generation Tool
Command Line Interface
"""
import sys
from typing import Optional

import click
from rich.console import Console

# Import main module
from .main import main_entry

console = Console()


@click.group(help="graphragkm - GraphRAG-driven AI Ontology Generation Tool")
def cli():
    """AIOntology command line tool entry point"""
    pass


@cli.command(name="run", help="Initialize and run ontology generation process")
@click.option(
    "--input-pdf",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    help="Input PDF file path",
)
def init_command(input_pdf: Optional[str]):
    """Initialize and run ontology generation process"""
    # Directly call main function
    main_entry(input_pdf)


@cli.command(name="version", help="Show version information")
def version():
    """Show version information"""
    from . import __version__

    console.print(f"[green]graphragkm version: {__version__}[/]")


def main():
    """Command line entry point"""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
