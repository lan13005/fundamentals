import glob
from pathlib import Path
import shutil
from rich.console import Console

console = Console()

# Cursor annoying takes forever to render .mdc files and setting vscode's files.associations"
#  to attempt rendering as .md file does not work. It still renders it as .mdc.
# This script swaps the .mdc and .md files whenever called

def main():
    md_files = glob.glob(".cursor/rules/*.md*")

    for file in md_files:
        if Path(file).suffix == ".md":
            shutil.move(file, file.replace(".md", ".mdc"))
            console.print(f"[yellow]{file}[/yellow] changed extension to [bold green].mdc[/bold green]")
        else:
            shutil.move(file, file.replace(".mdc", ".md"))
            console.print(f"[yellow]{file}[/yellow] changed extension to [bold green].md[/bold green]")
