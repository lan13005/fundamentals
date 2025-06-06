import json
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from fundamentals.utility.logging_config import get_logger

logger = get_logger(__name__)
console = Console()

# Data source: https://www.zillow.com/research/data/
ZILLOW_URLS = {
    "low": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv",
    "top": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv",
    "smoothed": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "single_family": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv",
}


def get_cache_filename(data_type: str) -> str:
    """Generate cache filename without date.

    Args:
        data_type: Type of data (low, top, smoothed, single_family)

    Returns:
        Cache filename
    """
    base_name = {
        "low": "City_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month",
        "top": "City_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month",
        "smoothed": "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month",
        "single_family": "City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month",
    }
    return f"{base_name[data_type]}.csv"


def download_zillow_data(data_type: str, cache_file: Path, force: bool = False) -> None:
    """Download Zillow data if not cached or if force is True.

    Args:
        data_type: Type of data to download
        cache_file: Path to cache file
        force: Whether to force download even if cache exists
    """
    if cache_file.exists() and not force:
        # Show file modification date
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        console.print(
            f"[green]Using cached data:[/green] {cache_file.name} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})"
        )
        console.print(
            "[dim]Note: Zillow updates data at the end of each month. Use --update-cache to force refresh.[/dim]"
        )
        return

    if force and cache_file.exists():
        console.print(f"[yellow]Removing cached file:[/yellow] {cache_file.name}")
        cache_file.unlink()

    url = ZILLOW_URLS[data_type]
    console.print(f"[blue]Downloading {data_type} data from Zillow...[/blue]")

    try:
        urllib.request.urlretrieve(url, cache_file)
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        console.print(f"[green]Downloaded:[/green] {cache_file.name} (created: {mod_time.strftime('%Y-%m-%d %H:%M')})")
        console.print("[dim]Zillow updates data at the end of each month.[/dim]")
    except Exception as e:
        console.print(f"[red]Error downloading {data_type} data: {e}[/red]")
        raise


def load_zillow_datasets(cache_dir: Path, force: bool = False) -> dict:
    """Load all Zillow datasets, downloading if necessary.

    Args:
        cache_dir: Directory for cache files
        force: Whether to force re-download

    Returns:
        Dictionary of DataFrames keyed by data type
    """
    datasets = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Zillow datasets...", total=len(ZILLOW_URLS))

        for data_type in ZILLOW_URLS.keys():
            cache_file = cache_dir / get_cache_filename(data_type)

            # Download if needed
            download_zillow_data(data_type, cache_file, force)

            # Load the data
            try:
                datasets[data_type] = pd.read_csv(cache_file)
                progress.advance(task)
            except Exception as e:
                console.print(f"[red]Error loading {data_type} data: {e}[/red]")
                raise

    return datasets


def filter_city_data(datasets: dict, city: str, state: str) -> dict:
    """Filter datasets for specific city and state.

    Args:
        datasets: Dictionary of DataFrames
        city: City name
        state: State abbreviation

    Returns:
        Dictionary of filtered DataFrames
    """
    filtered = {}

    for data_type, df in datasets.items():
        city_data = df.query(f"RegionName == '{city}' and StateName == '{state}'")
        if city_data.empty:
            console.print(f"[yellow]Warning: No data found for {city}, {state} in {data_type} dataset[/yellow]")
        filtered[data_type] = city_data

    return filtered


def load_housing_events(housing_data_dir: Path) -> list:
    """Load housing market inflection events from JSON file.

    Args:
        housing_data_dir: Directory containing housing data

    Returns:
        List of event dictionaries with date, event, and description
    """
    events_file = housing_data_dir / "city_data" / "usa_housing_inflections.json"

    if not events_file.exists():
        console.print(f"[yellow]Warning: Events file not found at {events_file}[/yellow]")
        return []

    try:
        with open(events_file) as f:
            events = json.load(f)
        console.print(f"[green]Loaded {len(events)} housing market events[/green]")
        return events
    except Exception as e:
        console.print(f"[red]Error loading events file: {e}[/red]")
        return []


def create_housing_plot(filtered_data: dict, city: str, state: str, output_dir: Path) -> None:
    """Create and save housing price plot with market events.

    Args:
        filtered_data: Dictionary of filtered DataFrames
        city: City name
        state: State abbreviation
        output_dir: Directory to save plot
    """
    # Color palette following styling guidelines
    colors = {
        "low": "#1f77b4",  # blue
        "top": "#d62728",  # red
        "smoothed": "#ff7f0e",  # orange
        "single_family": "#2ca02c",  # green
    }

    # Create figure with larger size for better text placement
    fig, ax = plt.subplots(figsize=(16, 10))

    # First pass: determine the maximum value of the "top" dataset for capping growth curves
    top_max_value = None
    if "top" in filtered_data and not filtered_data["top"].empty:
        top_data = filtered_data["top"]
        top_values = top_data.values[0][9:]  # Get values from column 9 onwards
        valid_top_values = [v for v in top_values if pd.notna(v)]
        if valid_top_values:
            top_max_value = max(valid_top_values)

    # Plot housing data
    date_range = None
    data_max = 0
    for data_type, data in filtered_data.items():
        if not data.empty:
            # Date columns start at index 9
            dates = pd.to_datetime(data.columns[9:])
            values = data.values[0][9:]

            # Filter out NaN values for both dates and values
            valid_mask = pd.notna(values)
            dates_clean = dates[valid_mask]
            values_clean = values[valid_mask]

            # Calculate growth factor for Zillow datasets
            if len(values_clean) > 1:
                growth_factor = values_clean[-1] / values_clean[0]
                label_with_growth = f"{data_type.replace('_', ' ').title()} ({growth_factor:.1f}x)"
            else:
                label_with_growth = data_type.replace("_", " ").title()

            ax.plot(dates_clean, values_clean, label=label_with_growth, color=colors[data_type], linewidth=2.5)

            # Calculate growth curve for the same date points as the data
            if len(dates_clean) > 1:

                def growth_curve_for_dates(annual_rate, N0, start_date, target_dates):
                    """Calculate growth curve values for specific dates."""
                    days_from_start = (target_dates - start_date).days
                    daily_growth_factor = (1 + annual_rate) ** (1 / 365)
                    prices = N0 * daily_growth_factor**days_from_start
                    return prices

                # Generate growth curve for the actual data dates
                if data_type == "low":
                    # Plot growth curves with capping logic
                    for rate, label, style in [
                        (0.03, "3% (Treasury Bills)", "solid"),
                        (0.06, "6% (Bonds)", "dotted"),
                        (0.10, "10% (S&P 500)", "dashed"),
                    ]:
                        growth_values = growth_curve_for_dates(rate, values_clean[0], dates_clean[0], dates_clean)

                        # Cap growth curve if it exceeds top dataset maximum
                        if top_max_value is not None:
                            # Find where growth curve first exceeds top max
                            valid_indices = growth_values <= top_max_value
                            if not valid_indices.all():
                                # Find the last valid index
                                last_valid_idx = np.where(valid_indices)[0]
                                if len(last_valid_idx) > 0:
                                    last_valid_idx = last_valid_idx[-1]
                                    # Only plot up to the last valid point
                                    capped_dates = dates_clean[: last_valid_idx + 1]
                                    capped_values = growth_values[: last_valid_idx + 1]
                                else:
                                    # All values exceed top max, don't plot this growth curve
                                    continue
                            else:
                                # No capping needed
                                capped_dates = dates_clean
                                capped_values = growth_values
                        else:
                            # No top data available, plot without capping
                            capped_dates = dates_clean
                            capped_values = growth_values

                        # Calculate growth factor for this curve
                        curve_growth_factor = growth_values[-1] / growth_values[0]
                        label_with_growth = f"{label} ({curve_growth_factor:.1f}x)"

                        # Set line properties based on style
                        linewidth = 3 if style == "dotted" else 2
                        ax.plot(
                            capped_dates,
                            capped_values,
                            label=label_with_growth,
                            color=colors[data_type],
                            linewidth=linewidth,
                            linestyle=style,
                            alpha=0.7,
                        )

            if date_range is None:
                date_range = (dates_clean.min(), dates_clean.max())

            # Track max value for line positioning
            max_val = max([v for v in values_clean if pd.notna(v)])
            data_max = max(data_max, max_val)

    # Load and add housing market events
    events = load_housing_events(Path("housing_data/"))

    if events and date_range:
        # Filter events to date range of data
        filtered_events = []
        for event in events:
            event_date = pd.to_datetime(event["date"])
            if date_range[0] <= event_date <= date_range[1]:
                filtered_events.append(event)

        # Add event annotations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Expand ylim to create space for text boxes within the figure
        text_space_ratio = 0.25  # 25% extra space at top for text
        new_y_max = y_max + (text_space_ratio * y_range)
        ax.set_ylim(y_min, new_y_max)

        # Recalculate range with new limits
        y_range = new_y_max - y_min

        # Create varied heights for text to avoid overlap
        num_events = len(filtered_events)
        if num_events > 0:
            # Create staggered heights within the expanded area
            height_positions = []
            # Start text area just above the original data area
            text_start = y_max + 0.02 * y_range
            text_height = (new_y_max - text_start) * 0.8  # Use 80% of available text space

            for i in range(num_events):
                # Create 4 different height levels, cycling through them
                level = i % 4
                height = text_start + (level * text_height / 4) + (text_height / 8)
                height_positions.append(height)

        for i, event in enumerate(filtered_events):
            event_date = pd.to_datetime(event["date"])

            # Draw dotted black line from near the top of original data down
            line_start_y = data_max + 0.02 * (y_max - y_min)  # Use original y_range for line start
            line_end_y = y_min

            ax.plot(
                [event_date, event_date],
                [line_start_y, line_end_y],
                color="black",
                linestyle=":",
                alpha=0.6,
                linewidth=1.5,
            )

            # Format event name with newlines for longer descriptions
            event_name = event["event"]
            if len(event_name) > 25:  # Break long event names
                words = event_name.split()
                mid_point = len(words) // 2
                event_name = " ".join(words[:mid_point]) + "\n" + " ".join(words[mid_point:])

            # Simple text annotation within the figure bounds
            text_y = height_positions[i]

            ax.text(
                event_date,
                text_y,
                event_name,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
                rotation=0,
            )

            # Add date label
            ax.text(
                event_date,
                line_start_y + 0.01 * (y_max - y_min),
                event_date.strftime("%Y-%m"),
                fontsize=7,
                ha="center",
                va="bottom",
                color="black",
                alpha=0.7,
            )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Home Value Index ($)", fontsize=12)
    ax.set_title(f"Zillow Home Value Index - {city}, {state}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="center left")

    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save plot with higher DPI for better quality
    output_file = output_dir / f"{city}_{state}_zillow_housing_with_events.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    console.print(f"[green]Enhanced plot with market events saved:[/green] {output_file}")
    if events:
        console.print(
            f"[blue]Included {len([e for e in events if date_range[0] <= pd.to_datetime(e['date']) <= date_range[1]])} market events in date range[/blue]"
        )


def print_summary_table(filtered_data: dict, city: str, state: str) -> None:
    """Print summary statistics table.

    Args:
        filtered_data: Dictionary of filtered DataFrames
        city: City name
        state: State abbreviation
    """
    table = Table(
        title=f"Housing Data Summary - {city}, {state}",
        box=box.ROUNDED,
        header_style="bold green",
    )
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Latest Value", style="yellow", justify="right")
    table.add_column("Months Recorded", style="yellow", justify="right")

    for data_type, data in filtered_data.items():
        if not data.empty:
            # Get latest non-null value
            latest_val = None
            records = 0
            for col in reversed(data.columns[9:]):
                val = data.iloc[0][col]
                if pd.notna(val):
                    latest_val = f"${val:,.0f}"
                    records = sum(pd.notna(data.iloc[0][9:]))
                    break

            table.add_row(data_type.replace("_", " ").title(), latest_val or "N/A", str(records))
        else:
            table.add_row(data_type.replace("_", " ").title(), "No Data", "0")

    console.print(table)


def run_zillow_housing_analysis(city: str = "Denver", state: str = "CO", update_cache: bool = False) -> None:
    """Run Zillow housing analysis for specified city and state.

    Args:
        city: City name
        state: State abbreviation
        update_cache: Whether to force cache update and re-download
    """
    console.rule("[bold blue]Zillow Housing Data Analysis")

    # Setup directories
    cache_dir = Path("housing_data")
    cache_dir.mkdir(exist_ok=True)

    # Print parameters
    params_table = Table(
        title="Analysis Parameters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="yellow", justify="right")
    params_table.add_row("City", city)
    params_table.add_row("State", state)
    params_table.add_row("Update Cache", str(update_cache))

    console.print(params_table)

    try:
        # Load datasets
        datasets = load_zillow_datasets(cache_dir, update_cache)

        # Filter for city/state
        filtered_data = filter_city_data(datasets, city, state)

        # Print summary
        print_summary_table(filtered_data, city, state)

        # Create plot
        create_housing_plot(filtered_data, city, state, cache_dir)

        explanation = f"""
        This analysis downloads and visualizes Zillow Home Value Index (ZHVI) data for {city}, {state}.
        Four different housing tiers are shown:

        • Low tier (0-33rd percentile): Most affordable homes
        • Top tier (67th-100th percentile): Most expensive homes
        • Smoothed tier (33rd-67th percentile): Middle market homes
        • Single family homes: Detached single-family residences

        Data is cached locally and will be reused unless --update-cache is specified.
        Zillow updates their data at the end of each month.
        """

        console.print(Panel(explanation.strip(), title="Explanation", border_style="blue"))

    except Exception as e:
        console.print(f"[red]Error in housing analysis: {e}[/red]")
        raise
