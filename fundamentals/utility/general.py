import datetime as dt
import re
from typing import Optional

import matplotlib as mpl
import requests
from bs4 import BeautifulSoup
from cycler import cycler

from fundamentals.utility.logger import get_logger

logger = get_logger(__name__)


def get_sp500_tickers() -> list[str]:
    """Get latest S&P 500 tickers from Wikipedia."""
    wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
    table = BeautifulSoup(wiki, "lxml").find("table", {"id": "constituents"})
    tickers = [row.find_all("td")[0].text.strip() for row in table.find_all("tr")[1:]]
    tickers = [t.replace(".", "-") for t in tickers]  # BRK.B -> BRK-B, this is done for yfinance
    return tickers


def get_nasdaq_tickers() -> list[str]:
    """Get the NASDAQ tickers from Nasdaq."""
    wiki = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt").text
    tickers = [t.split("|")[0] for t in wiki.split("\n")[1:] if len(t) > 0]
    tickers = [t.replace(".", "-") for t in tickers]  # BRK.B -> BRK-B, this is done for yfinance
    return tickers


def get_latest_quarter_end(today: Optional[dt.date] = None) -> dt.date:
    """Return the most recent completed financial quarter end date relative to today."""
    today = today or dt.date.today()
    year = today.year
    if today.month >= 10:
        return dt.date(year, 9, 30)
    elif today.month >= 7:
        return dt.date(year, 6, 30)
    elif today.month >= 4:
        return dt.date(year, 3, 31)
    else:
        return dt.date(year - 1, 12, 31)


def clean_value_for_markdown_cell(cell_text: str) -> str:
    """Cleans a single data cell's text content for Markdown output."""
    text = cell_text.strip()
    if not text:
        return ""

    # 1. Remove currency symbols and commas first.
    # This helps simplify the parenthetical check for various formats like ($1,234) or $(1,234)
    text_no_symbols = text.replace("$", "").replace(",", "")

    is_negative = False
    # 2. Check for parentheses indicating a negative number on the symbol-cleaned text
    if text_no_symbols.startswith("(") and text_no_symbols.endswith(")"):
        is_negative = True
        # Get the content inside the parentheses
        text_core = text_no_symbols[1:-1].strip()
    else:
        # If not in parentheses, use the symbol-cleaned text directly
        text_core = text_no_symbols.strip()

    if not text_core:  # If after all stripping, text is empty
        return ""

    # 3. Determine if the core text is a number candidate
    is_number_candidate = False
    # Allow for one leading optional minus sign for the core content
    temp_for_check = text_core
    if temp_for_check.startswith("-"):
        temp_for_check = temp_for_check[1:]

    if temp_for_check.count(".") <= 1:  # Allow zero or one decimal point
        if temp_for_check.replace(".", "", 1).isdigit():  # Check if all remaining are digits
            is_number_candidate = True

    # 4. Apply negative sign or return original-like content
    if is_number_candidate:
        if is_negative:
            # Prepend '-' if it's not already negative from the core content (e.g. "(-5)" -> "-5")
            if text_core.startswith("-"):
                return text_core  # Already correctly negative
            else:
                return "-" + text_core
        else:
            return text_core  # Positive number or already correctly signed negative number
    else:
        # If it was in parentheses but not a number (e.g., "(Adjusted)"), return the original cell text,
        # stripped of symbols but perhaps with parens if meaningful.
        # For simplicity and to avoid misinterpreting non-numeric parenthesized text,
        # returning the original stripped cell text is safest if it's not a clear number.
        # However, if it was `(abc)`, `text_core` is `abc`. If it was `($abc)`, `text_core` is `abc`.
        # If it was `$(abc)`, `text_core` is `abc`.
        # The current logic will return `abc`. If the original formatting `(abc)` is desired for
        # non-numeric parenthetical data, this would need adjustment.
        # For now, we assume if it's not a number, we return its core content.
        return text_core


def reformat_markdown_financial_table(markdown_text_input: str) -> str:
    """
    Reformats a Markdown string containing a financial table to be more
    LLM-readable while remaining in Markdown format.
    Strips ANSI codes and cleans financial data.
    """

    ansi_escape_pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    # Process line by line for ANSI removal and non-breaking space replacement
    processed_lines = []
    for line in markdown_text_input.split("\n"):
        line_no_ansi = ansi_escape_pattern.sub("", line)
        line_no_nbsp = line_no_ansi.replace("\u00a0", " ")
        processed_lines.append(line_no_nbsp)

    output_formatted_lines = []
    table_column_headers_content = []

    for _, current_line in enumerate(processed_lines):
        # Strip the line after ANSI/NBSP cleaning for processing logic
        stripped_current_line = current_line.strip()

        if not stripped_current_line:  # Handle blank lines
            if output_formatted_lines and output_formatted_lines[-1].strip() != "":
                output_formatted_lines.append("")
            continue

        # --- Title and Subtitle Processing ---
        if stripped_current_line.startswith("##"):
            output_formatted_lines.append(f"## {stripped_current_line.lstrip('# ').strip()}")
            table_column_headers_content = []
        elif (
            stripped_current_line.startswith("**")
            and "*" in stripped_current_line
            and "(" in stripped_current_line
            and ")" in stripped_current_line
        ):
            # Use original current_line (before strip) for regex if spaces are part of pattern
            match = re.match(r"\s*\*\*(.*?)\*\*\s*\*\((.*?)\)\*\s*", current_line)
            if match:
                bold_part = match.group(1).strip()
                italic_part = match.group(2).strip()
                output_formatted_lines.append(f"**{bold_part}** *({italic_part})*")
            else:
                output_formatted_lines.append(stripped_current_line)  # Fallback
            table_column_headers_content = []

        # --- Table Row Processing ---
        elif stripped_current_line.startswith("|") and stripped_current_line.endswith("|"):
            # Use current_line (not stripped_current_line) for splitting to preserve internal spaces for cell stripping
            cells_from_split = current_line.split("|")
            if len(cells_from_split) < 2:
                output_formatted_lines.append(stripped_current_line)
                table_column_headers_content = []
                continue

            current_row_cell_contents = [
                cell.strip() for cell in cells_from_split[1:-1]
            ]  # Strip each cell individually

            is_separator_line = False
            if current_row_cell_contents and all(
                re.fullmatch(r"-+\s*", cell_content.strip())
                for cell_content in current_row_cell_contents
                if cell_content.strip()
            ):
                if any(cell_content.strip() for cell_content in current_row_cell_contents):
                    is_separator_line = True

            if is_separator_line:
                num_columns = (
                    len(table_column_headers_content)
                    if table_column_headers_content
                    else len(current_row_cell_contents)
                )
                separator_cells = [" --- "] * num_columns
                output_formatted_lines.append(f"|{'|'.join(separator_cells)}|")

            elif (
                not table_column_headers_content
                and any(date_keyword in stripped_current_line for date_keyword in ["Mar", "Dec", "Jun", "Jul", "Apr"])
                and "$" not in stripped_current_line
                and "(" not in stripped_current_line
            ):  # Heuristic for header
                table_column_headers_content = [cell.strip() for cell in current_row_cell_contents]
                formatted_header_cells = [f" {name} " for name in table_column_headers_content]
                output_formatted_lines.append(f"|{'|'.join(formatted_header_cells)}|")

            elif not table_column_headers_content and current_row_cell_contents:  # Data before header or unknown table
                cleaned_cells = [clean_value_for_markdown_cell(cell) for cell in current_row_cell_contents]
                padded_cells = [f" {data} " for data in cleaned_cells]
                output_formatted_lines.append(f"|{'|'.join(padded_cells)}|")
            elif current_row_cell_contents:  # Ensure there's content to process
                item_name = current_row_cell_contents[0].strip()  # Item name always stripped
                values = current_row_cell_contents[1:]
                cleaned_values = [clean_value_for_markdown_cell(val) for val in values]

                all_cells_for_this_row = [f" {item_name} "] + [f" {cv} " for cv in cleaned_values]

                if table_column_headers_content:  # Align with header if present
                    expected_cols = len(table_column_headers_content)
                    current_cols = len(all_cells_for_this_row)
                    if current_cols < expected_cols:
                        all_cells_for_this_row.extend(["  "] * (expected_cols - current_cols))
                    elif current_cols > expected_cols:
                        all_cells_for_this_row = all_cells_for_this_row[:expected_cols]

                output_formatted_lines.append(f"|{'|'.join(all_cells_for_this_row)}|")
                # else: it's an empty | | situation or similar, effectively skipped if not matching other patterns

        else:  # Non-matching lines (e.g., other text from ANSI stripping, or reset context)
            # If a line became empty after ANSI stripping, it will be handled by the initial `if not stripped_current_line:`
            if stripped_current_line:  # Only add if it has content after stripping
                output_formatted_lines.append(stripped_current_line)
            table_column_headers_content = []

    final_output = "\n".join(output_formatted_lines)
    while final_output.endswith("\n\n"):
        final_output = final_output[:-1]
    if markdown_text_input.strip() and not final_output.endswith("\n") and final_output.strip():
        final_output += "\n"
    elif not markdown_text_input.strip() and final_output == "\n":
        final_output = ""
    return final_output


def update_plot_style():
    """
    Apply pastel color palette, no grids
    """
    pastel_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:grey",
        "tab:pink",
        "tab:brown",
        "tab:purple",
    ]

    mpl.rcParams.update(
        {
            # Colors
            "axes.prop_cycle": cycler("color", pastel_colors),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # Spines
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            # Ticks
            "xtick.bottom": True,
            "xtick.top": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "sans-serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            # Lines & markers
            "lines.linewidth": 1.2,
            "lines.markersize": 6,
            "markers.fillstyle": "full",
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
            # Disable grid
            "axes.grid": False,
        }
    )
