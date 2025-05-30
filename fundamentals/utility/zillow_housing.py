import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

# Data source: https://www.zillow.com/research/data/
ZILLOW_URLS = {
    "low": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv",
    "top": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv",
    "smoothed": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "single_family": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
}


def get_cache_filename(data_type: str, date_str: str) -> str:
    """Generate cache filename with date appended.
    
    Args:
        data_type: Type of data (low, top, smoothed, single_family)
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Cache filename with date appended
    """
    base_name = {
        "low": "City_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month",
        "top": "City_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month",
        "smoothed": "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month",
        "single_family": "City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month"
    }
    return f"{base_name[data_type]}_{date_str}.csv"


def download_zillow_data(data_type: str, cache_file: Path, force: bool = False) -> None:
    """Download Zillow data if not cached or if force is True.
    
    Args:
        data_type: Type of data to download
        cache_file: Path to cache file
        force: Whether to force download even if cache exists
    """
    if cache_file.exists() and not force:
        console.print(f"[green]Using cached data:[/green] {cache_file.name}")
        return
        
    if force and cache_file.exists():
        console.print(f"[yellow]Removing cached file:[/yellow] {cache_file.name}")
        cache_file.unlink()
    
    url = ZILLOW_URLS[data_type]
    console.print(f"[blue]Downloading {data_type} data from Zillow...[/blue]")
    
    try:
        urllib.request.urlretrieve(url, cache_file)
        console.print(f"[green]Downloaded:[/green] {cache_file.name}")
    except Exception as e:
        console.print(f"[red]Error downloading {data_type} data: {e}[/red]")
        raise


def load_zillow_datasets(cache_dir: Path, date_str: str, force: bool = False) -> dict:
    """Load all Zillow datasets, downloading if necessary.
    
    Args:
        cache_dir: Directory for cache files
        date_str: Date string for cache filenames
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
            cache_file = cache_dir / get_cache_filename(data_type, date_str)
            
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


def create_housing_plot(filtered_data: dict, city: str, state: str, output_dir: Path) -> None:
    """Create and save housing price plot.
    
    Args:
        filtered_data: Dictionary of filtered DataFrames
        city: City name
        state: State abbreviation  
        output_dir: Directory to save plot
    """
    # Color palette following styling guidelines
    colors = {
        "low": "#1f77b4",          # blue
        "top": "#d62728",          # red
        "smoothed": "#ff7f0e",     # orange
        "single_family": "#2ca02c" # green
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for data_type, data in filtered_data.items():
        if not data.empty:
            # Date columns start at index 9
            dates = pd.to_datetime(data.columns[9:])
            values = data.values[0][9:]
            
            ax.plot(dates, values, 
                   label=data_type.replace("_", " ").title(),
                   color=colors[data_type],
                   linewidth=2)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Home Value Index ($)", fontsize=12)
    ax.set_title(f"Zillow Home Value Index - {city}, {state}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{city}_{state}_zillow_housing.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    console.print(f"[green]Plot saved:[/green] {output_file}")


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
    table.add_column("Records", style="yellow", justify="right")
    
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
            
            table.add_row(
                data_type.replace("_", " ").title(),
                latest_val or "N/A",
                str(records)
            )
        else:
            table.add_row(
                data_type.replace("_", " ").title(),
                "No Data",
                "0"
            )
    
    console.print(table)


def run_zillow_housing_analysis(city: str = "Denver", state: str = "CO", ignore_cache: bool = False) -> None:
    """Run Zillow housing analysis for specified city and state.
    
    Args:
        city: City name
        state: State abbreviation
        ignore_cache: Whether to ignore existing cache and re-download
    """
    console.rule("[bold blue]Zillow Housing Data Analysis")
    
    # Setup directories
    cache_dir = Path("housing_data")
    cache_dir.mkdir(exist_ok=True)
    
    # Use current date for cache files
    date_str = datetime.now().strftime("%Y-%m-%d")
    
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
    params_table.add_row("Cache Date", date_str)
    params_table.add_row("Ignore Cache", str(ignore_cache))
    
    console.print(params_table)
    
    try:
        # Load datasets
        datasets = load_zillow_datasets(cache_dir, date_str, ignore_cache)
        
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
        
        Data is cached locally with today's date and will be reused unless --ignore-cache is specified.
        """
        
        console.print(
            Panel(explanation.strip(), title="Explanation", border_style="blue")
        )
        
    except Exception as e:
        console.print(f"[red]Error in housing analysis: {e}[/red]")
        raise
