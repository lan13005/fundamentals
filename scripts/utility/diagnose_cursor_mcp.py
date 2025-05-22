import os
import glob
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

LOG_ROOT = os.path.expanduser("~/Library/Application Support/Cursor/logs/")
LOG_FILENAME = "Cursor MCP.log"

ERROR_PATTERNS = [
    re.compile(r"\\[error\\]", re.IGNORECASE),
    re.compile(r"Unexpected token.*not valid JSON", re.IGNORECASE),
    re.compile(r"Error connecting to MCP server", re.IGNORECASE),
    re.compile(r"Connection refused", re.IGNORECASE),
    re.compile(r"Failed to execute tool", re.IGNORECASE),
]
SUCCESS_PATTERN = re.compile(r"Successfully called tool", re.IGNORECASE)

console = Console()

def find_log_files(root: str, log_filename: str):
    """
    Find all log files named log_filename under root, sorted by mtime descending.
    Returns a list of (path, mtime) tuples.
    """
    log_files = []
    for dirpath, _, _ in os.walk(root):
        candidate = os.path.join(dirpath, log_filename)
        if os.path.isfile(candidate):
            mtime = os.path.getmtime(candidate)
            log_files.append((candidate, mtime))
    log_files.sort(key=lambda x: x[1], reverse=True)
    return log_files

def analyze_log_file(path: str):
    """
    Analyze a log file for errors and successes.
    Returns a dict with status, error_lines, success_lines, and context.
    """
    error_lines = []
    success_lines = []
    context_lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if any(p.search(line) for p in ERROR_PATTERNS):
                error_lines.append((i, line.strip()))
                # Add context: 2 lines before and after
                context = lines[max(0, i-2):min(len(lines), i+3)]
                context_lines.append((i, [l.strip() for l in context]))
            if SUCCESS_PATTERN.search(line):
                success_lines.append((i, line.strip()))
        status = "error" if error_lines else ("success" if success_lines else "unknown")
        return {
            "status": status,
            "error_lines": error_lines,
            "success_lines": success_lines,
            "context": context_lines,
        }
    except Exception as e:
        return {"status": "read_error", "error": str(e)}

def print_log_report(logs):
    """
    Print a summary report of analyzed logs using rich.
    """
    for log in logs:
        path = log["path"]
        analysis = log["analysis"]
        status = analysis["status"]
        path_text = Text(path, style="bold cyan")
        
        paths_with_errors = []
        
        if status == "error":
            console.print(Panel.fit(path_text, title="[red]ERROR DETECTED[/red]", border_style="red", box=box.ROUNDED))
            paths_with_errors.append(path)
            for idx, line in analysis["error_lines"]:
                console.print(f"[red]Error line {idx+1}:[/red] {line}")
            for idx, context in analysis["context"]:
                console.print(Panel("\n".join(context), title=f"Context around error line {idx+1}", border_style="yellow"))
        elif status == "success":
            console.print(Panel.fit(path_text, title="[green]Success Entries Found[/green]", border_style="green", box=box.ROUNDED))
            for idx, line in analysis["success_lines"]:
                console.print(f"[green]Success line {idx+1}:[/green] {line}")
        elif status == "unknown":
            console.print(Panel.fit(path_text, title="[yellow]No errors or successes found[/yellow]", border_style="yellow", box=box.ROUNDED))
        else:
            console.print(Panel.fit(path_text, title="[red]Log Read Error[/red]", border_style="red", box=box.ROUNDED))
            console.print(f"[red]Error reading log:[/red] {analysis.get('error')}")
        console.print("\n")
        
    if paths_with_errors:
        for path in paths_with_errors:
            escaped_path = path.replace("\s+", " ").replace(" ", "\\ ")
            console.print(f"[red]Path:[/red] {escaped_path}")

def parse_range_arg(range_arg: str, total: int):
    """
    Parse a range argument like '0', '0:5', '5:', ':5' and return (start, end) indices.
    """
    if not range_arg:
        return 0, total
    if ':' in range_arg:
        parts = range_arg.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {range_arg}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else total
    else:
        start = int(range_arg)
        end = start + 1
    # Clamp to valid range
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    return start, end

def main(range_arg: str = None):
    """
    Diagnose Cursor MCP logs for errors and successes.
    Args:
        range_arg (str or None): Range of logs to analyze: N, N:M, N:, :M. If None, all logs are analyzed.
    """
    console.rule("[bold blue]Cursor MCP Log Diagnostics[/bold blue]")
    if not os.path.isdir(LOG_ROOT):
        console.print(f"[red]Log root directory not found:[/red] {LOG_ROOT}")
        return
    log_files = find_log_files(LOG_ROOT, LOG_FILENAME)
    if not log_files:
        console.print(f"[yellow]No '{LOG_FILENAME}' files found in {LOG_ROOT}[/yellow]")
        return
    total = len(log_files)
    start, end = parse_range_arg(range_arg, total)
    selected_log_files = log_files[start:end]
    logs = []
    for path, mtime in selected_log_files:
        analysis = analyze_log_file(path)
        logs.append({"path": path, "mtime": mtime, "analysis": analysis})
    print_log_report(logs)
