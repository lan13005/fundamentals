"""
Discounted Cash Flow (DCF) modeling framework.

This module provides a base class for implementing DCF models with Monte Carlo simulation
capabilities for uncertainty quantification.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import corner
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from scipy import stats
from scipy.stats import gaussian_kde, norm

from fundamentals.utility.logger import get_logger

logger = get_logger(__name__)
console = Console()


class BaseDCFModel(ABC):
    """
    Abstract base class for Discounted Cash Flow models.

    This class provides the framework for implementing DCF models with:
    - Abstract model definition
    - Prior distribution specification
    - Monte Carlo sampling with correlation support
    - Uncertainty quantification
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DCF model with financial data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Financial data to be analyzed, must contain required columns
        """
        self.df = df
        self._original_df = df.copy()  # Store original for set_center method
        self.distributions = {}
        self.parameter_names = []
        self._correlation_matrix = None

    def set_center(self, center_idx: int) -> None:
        """
        Set the center date for sliding window analysis.
        
        This method subsets the data up to the center date and reconfigures
        priors that depend on historical data (like empirical growth rates).
        
        Parameters:
        -----------
        center_idx : int
            Index of the center date (data will be subset to [:center_idx+1])
        """
        if center_idx >= len(self._original_df):
            raise ValueError(f"Center index {center_idx} exceeds original data length {len(self._original_df)}")
        
        if center_idx < 0:
            raise ValueError(f"Center index {center_idx} must be non-negative")
        
        # Subset data up to center date
        self.df = self._original_df.iloc[:center_idx + 1].copy()
        
        # Reconfigure priors with new data subset if we have original distributions stored
        if hasattr(self, '_original_distributions') and hasattr(self, '_original_correlation_matrix'):
            self.configure_priors(self._original_distributions, self._original_correlation_matrix)
        else:
            raise ValueError("Original distributions not stored. Cannot reconfigure priors for sliding analysis.")

    def _get_correlation_matrix(self) -> np.ndarray:
        """
        Define correlation matrix for parameters when using correlated sampling.
        Default implementation returns identity matrix (independent sampling).
        Override in subclasses for correlated sampling.

        Returns:
        --------
        np.ndarray : Correlation matrix (symmetric, positive definite)
        """
        if self._correlation_matrix is not None:
            return self._correlation_matrix

        n_params = len(self.parameter_names)
        return np.eye(n_params)

    def _check_configuration(self) -> None:
        """
        Check if model is properly configured before running simulations.
        
        Raises:
        -------
        ValueError : If distributions are not configured or correlation matrix is invalid
        """
        if not hasattr(self, "distributions"):
            raise ValueError("Prior distributions not configured. Call configure_priors() first.")
        
        if not self.parameter_names:
            raise ValueError("No parameters defined. Call configure_priors() first.")
        
                # Check if we have distributions for all parameters
        if not self.distributions:
            raise ValueError("No distributions configured. Call configure_priors() first.")
        
        missing_distributions = [p for p in self.parameter_names if p not in self.distributions]
        if missing_distributions:
            raise ValueError(f"Missing distributions for parameters: {missing_distributions}")
        
        # Check correlation matrix if provided
        if self._correlation_matrix is not None:
            n_params = len(self.parameter_names)
            if self._correlation_matrix.shape != (n_params, n_params):
                raise ValueError(
                    f"Correlation matrix shape {self._correlation_matrix.shape} doesn't match number of parameters {n_params}"
                )
            
            # Check if matrix is symmetric and positive semidefinite
            if not np.allclose(self._correlation_matrix, self._correlation_matrix.T):
                logger.warning("Correlation matrix is not symmetric")
            
            eigenvals = np.linalg.eigvals(self._correlation_matrix)
            if np.any(eigenvals < -1e-8):
                logger.warning("Correlation matrix may not be positive semidefinite")

    @abstractmethod
    def configure_priors(self, distributions: Dict[str, Any], correlation_matrix: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Configure prior distributions and correlation matrix for the model.

        Parameters:
        -----------
        distributions : dict
            Dictionary mapping parameter names to scipy.stats distributions
        correlation_matrix : np.ndarray, optional
            Correlation matrix for parameters. If None, uses identity matrix (independent sampling)
        **kwargs : additional arguments
            Additional arguments for subclass-specific configuration
        """
        pass

    @abstractmethod
    def calculate_dcf(self, **params) -> Tuple[float, float, float]:
        """
        Calculate DCF value and implied stock price for given parameters.

        Parameters:
        -----------
        **params : keyword arguments
            Model parameters (discount_rate_scale, growth_rate, etc.)

        Returns:
        --------
        tuple : (pv_fcf_fraction, dcf_value, stock_price)
        """
        pass

    @abstractmethod
    def create_summary_table(self, **params) -> Table:
        """
        Create Rich table summarizing model parameters and results.

        Parameters:
        -----------
        **params : keyword arguments
            Parameters and results to display in the table

        Returns:
        --------
        Table : Rich table object
        """
        pass

    def simulate(
        self,
        n_samples: int = 1000,
        random_state: int = 42,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Run Monte Carlo simulation for DCF valuation with parameter sampling.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        random_state : int
            Random seed for reproducibility
        show_progress : bool
            Whether to show progress bars (default=True)

        Returns:
        --------
        tuple : (pv_fcf_fractions, dcf_values, stock_prices, parameter_samples)
            - pv_fcf_fractions : np.ndarray of shape (n_samples,)
            - dcf_values : np.ndarray of shape (n_samples,)
            - stock_prices : np.ndarray of shape (n_samples,)
            - parameter_samples : Dict[str, np.ndarray] with parameter samples
        """
        self._check_configuration()

        if show_progress:
            console.rule("[bold blue]Monte Carlo DCF Simulation")
            logger.info("Starting Monte Carlo DCF simulation")

        # Sample parameters
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                results = self._run_simulation(n_samples, random_state, progress)
        else:
            results = self._run_simulation(n_samples, random_state, None)

        pv_fcf_fractions, dcf_values, stock_prices, parameter_samples = results

        if show_progress:
            logger.info("Results returned with shapes:")
            logger.debug(f" - pv_fcf_fractions: {pv_fcf_fractions.shape}")
            logger.debug(f" - dcf_values: {dcf_values.shape}")
            logger.debug(f" - stock_prices: {stock_prices.shape}")
            logger.debug(f" - parameter_samples keys: {list(parameter_samples.keys())}")
            logger.info(f"Simulation complete: {n_samples:,} samples generated")

        return pv_fcf_fractions, dcf_values, stock_prices, parameter_samples

    def _run_simulation(
        self,
        n_samples: int,
        random_state: int,
        progress: Optional[Progress],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Internal method to run the actual simulation logic."""
        rng = np.random.default_rng(random_state)

        # Get correlation matrix
        corr = self._get_correlation_matrix()

        n_params = len(self.parameter_names)

        if corr.shape != (n_params, n_params):
            raise ValueError(f"Correlation matrix shape {corr.shape} doesn't match number of parameters {n_params}")

        # Gaussian copula
        L = np.linalg.cholesky(corr)
        Z = rng.standard_normal(size=(n_samples, n_params))
        correlated_Z = Z @ L.T
        U = norm.cdf(correlated_Z)  # uniform samples with preserved rank correlation

        if progress:
            sample_task = progress.add_task("Sampling parameters...", total=100)
            progress.update(sample_task, advance=50)

        # Transform to marginal distributions
        samples = {}
        for i, param_name in enumerate(self.parameter_names):
            distribution = self.distributions[param_name]
            samples[param_name] = distribution.ppf(U[:, i])

        # Apply constraints with rejection sampling
        samples = self._apply_parameter_constraints(samples, rng, L)

        if progress:
            progress.update(sample_task, advance=50)

        # Task 2: Run DCF simulations
        if progress:
            sim_task = progress.add_task("Running DCF simulations...", total=n_samples)

        # Initialize arrays to store results
        pv_fcf_fractions = np.zeros(n_samples)
        dcf_values = np.zeros(n_samples)
        stock_prices = np.zeros(n_samples)

        for i in range(n_samples):
            # Extract parameters for this iteration
            params = {name: samples[name][i] for name in self.parameter_names}

            # Calculate DCF for this parameter set
            pv_fcf_fraction, dcf_val, stock_price = self.calculate_dcf(**params)
            pv_fcf_fractions[i] = pv_fcf_fraction
            dcf_values[i] = dcf_val
            stock_prices[i] = stock_price

            if progress and i % max(1, n_samples // 100) == 0:  # Update every 1%
                progress.update(sim_task, advance=max(1, n_samples // 100))

        if progress:
            # Print summary
            logger.info(f"PV FCF Fractions: {pv_fcf_fractions.mean():.2f} ± {pv_fcf_fractions.std():.2f}")
            logger.info(f"DCF values: {dcf_values.mean():.2f} ± {dcf_values.std():.2f}")
            logger.info(f"Stock prices: {stock_prices.mean():.2f} ± {stock_prices.std():.2f}")

        return pv_fcf_fractions, dcf_values, stock_prices, samples

    def sliding_dcf_analysis(
        self,
        cagr_lookback: float,
        forecast_horizon: int,
        n_samples: int = 1000,
        random_state: int = 42,
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Perform sliding window DCF analysis with Monte Carlo simulation across all available time periods.
        
        This function slides a DCF calculation window across the entire time series,
        where each position represents a different "center date" for the analysis.
        For each center date, runs a full Monte Carlo simulation generating n_samples.
        The center date must have sufficient historical data for CAGR calculation
        and can forecast into the future up to forecast_horizon periods.

        Parameters:
        -----------
        cagr_lookback : float
            Number of years to look back for CAGR calculation
        forecast_horizon : int
            Number of years to forecast forward for DCF calculation
        n_samples : int
            Number of Monte Carlo samples to generate per center date (default=1000)
        random_state : int
            Random seed for reproducibility (default=42)
        output_file : str, optional
            If provided, save results to CSV file

        Returns:
        --------
        pd.DataFrame : Results with columns:
            - 'center_date': The center date for this analysis
            - 'sample_id': Sample number within each center date
            - 'stock_price': Simulated stock price
            - 'dcf_value': Simulated DCF value
            - 'pv_fcf_fraction': Simulated PV FCF fraction
            - Plus all parameter samples (discount_rate_scale, growth_rate, etc.)
        """
        self._check_configuration()
            
        console.rule("[bold blue]Sliding DCF Monte Carlo Analysis")
        logger.info("Starting sliding DCF Monte Carlo analysis")
        logger.info(f"CAGR Horizon: {cagr_lookback} years ({cagr_lookback * 4:.0f} quarters)")
        logger.info(f"Forecast Horizon: {forecast_horizon} years")
        logger.info(f"Samples per center date: {n_samples:,}")
        
        # Ensure df has datetime index
        if not isinstance(self._original_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for sliding analysis")
        
        results = []
        quarters_needed = int(cagr_lookback * 4)  # Convert years to quarters
        forecast_quarters = int(forecast_horizon * 4)  # Convert forecast horizon to quarters
        
        # Find valid center dates
        # Start: Must have enough historical data for CAGR calculation
        valid_start_idx = quarters_needed
        # End: Can go to the end of data since we can forecast into the future
        valid_end_idx = len(self._original_df)
        
        total_center_dates = valid_end_idx - valid_start_idx
        total_samples = total_center_dates * n_samples
        
        # Debug information
        logger.debug(f"Data range: {self._original_df.index[0]} to {self._original_df.index[-1]}")
        logger.debug(f"Total data points: {len(self._original_df)}")
        logger.debug(f"CAGR horizon: {cagr_lookback} years ({quarters_needed} quarters)")
        logger.debug(f"Forecast horizon: {forecast_horizon} years ({forecast_quarters} quarters)")
        logger.info(f"Valid center date range: {self._original_df.index[valid_start_idx]} to {self._original_df.index[valid_end_idx-1]}")
        logger.info(f"Total center dates: {total_center_dates}")
        logger.info(f"Total samples to generate: {total_samples:,}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Running sliding Monte Carlo analysis...", total=total_center_dates)
            
            for center_idx in range(valid_start_idx, valid_end_idx):
                center_date = self._original_df.index[center_idx]
                
                try:
                    # Set center date for sliding window analysis
                    self.set_center(center_idx)
                    
                    # Run Monte Carlo simulation for this center date
                    # Use different random seed per center date for proper randomization
                    pv_fcf_fractions, dcf_values, stock_prices, parameter_samples = self.simulate(
                        n_samples=n_samples,
                        random_state=random_state + center_idx,
                        show_progress=False  # Disable nested progress to avoid conflicts
                    )
                    
                    # Extract results and add center date info
                    for sample_idx in range(n_samples):
                        sample_result = {
                            'center_date': center_date,
                            'sample_id': sample_idx,
                            'stock_price': stock_prices[sample_idx],
                            'dcf_value': dcf_values[sample_idx],
                            'pv_fcf_fraction': pv_fcf_fractions[sample_idx],
                        }
                        
                        # Add all parameter samples
                        for param_name, param_values in parameter_samples.items():
                            sample_result[param_name] = param_values[sample_idx]
                        
                        results.append(sample_result)
                        
                except Exception as e:
                    logger.warning(f"Failed to simulate DCF for {center_date}: {e}")
                    continue
                
                progress.advance(task)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Check if we have any results
        if len(results_df) == 0:
            logger.error("No successful simulations! Check your data and parameters.")
            return results_df
        
        # Save to file if requested
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")
        
        # Print summary statistics
        console.print("\n[bold blue]Summary Statistics:[/bold blue]")
        summary_table = Table(show_header=True, box=box.SIMPLE)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Samples", f"{len(results_df):,}")
        summary_table.add_row("Center Dates", f"{results_df['center_date'].nunique():,}")
        summary_table.add_row("Samples per Date", f"{n_samples:,}")
        summary_table.add_row("Date Range", f"{results_df['center_date'].min().strftime('%Y-%m-%d')} to {results_df['center_date'].max().strftime('%Y-%m-%d')}")
        summary_table.add_row("Avg Stock Price", f"${results_df['stock_price'].mean():.2f}")
        summary_table.add_row("Stock Price Std", f"${results_df['stock_price'].std():.2f}")
        summary_table.add_row("Min Stock Price", f"${results_df['stock_price'].min():.2f}")
        summary_table.add_row("Max Stock Price", f"${results_df['stock_price'].max():.2f}")
        
        # Add parameter statistics
        param_columns = [col for col in results_df.columns if col not in ['center_date', 'sample_id', 'stock_price', 'dcf_value', 'pv_fcf_fraction']]
        summary_table.add_row("Parameter Columns", f"{len(param_columns)}")
        
        console.print(summary_table)
        logger.info("Sliding Monte Carlo DCF analysis complete")
        logger.debug(f"Result DataFrame shape: {results_df.shape}")
        logger.debug(f"Columns: {list(results_df.columns)}")
        
        return results_df

    def _create_constraint_enforcer(self, constraint_func, constraint_name: str):
        """
        Create a constraint validation function for rejection sampling.
        
        Parameters:
        -----------
        constraint_func : callable
            Function that takes samples dict and returns boolean mask of valid samples
        constraint_name : str
            Name of constraint for logging
            
        Returns:
        --------
        callable : Constraint validation function
        """
        def enforcer(samples, rng, L):
            valid_mask = constraint_func(samples)
            iteration = 0
            max_iterations = 100
            
            while not np.all(valid_mask) and iteration < max_iterations:
                # Resample only invalid entries
                n_invalid = np.sum(~valid_mask)
                Z_new = rng.standard_normal(size=(n_invalid, len(self.parameter_names))) @ L.T
                U_new = norm.cdf(Z_new)

                # Replace invalid entries
                idxs = np.where(~valid_mask)[0]
                for i, param_name in enumerate(self.parameter_names):
                    distribution = self.distributions[param_name]
                    new_values = distribution.ppf(U_new[:, i])
                    samples[param_name][idxs] = new_values

                valid_mask = constraint_func(samples)
                iteration += 1
                
            if iteration >= max_iterations:
                logger.warning(f"{constraint_name} constraint couldn't be satisfied after {max_iterations} iterations")
                
            return samples
        return enforcer

    def _apply_parameter_constraints(
        self, samples: Dict[str, np.ndarray], rng: np.random.Generator, L: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Apply parameter constraints using rejection sampling.
        Default implementation ensures terminal_growth < discount_rate_scale (assuming WACC > 0).
        Override in subclasses for additional constraints.
        """
        if "terminal_growth" not in samples or "discount_rate_scale" not in samples:
            return samples

        # Define constraint: terminal growth should be less than discount rate scale
        def terminal_growth_constraint(samples_dict):
            return samples_dict["terminal_growth"] < samples_dict["discount_rate_scale"]
        
        # Apply constraint using the enforcer
        constraint_enforcer = self._create_constraint_enforcer(
            terminal_growth_constraint,
            "terminal_growth < discount_rate_scale"
        )
        
        return constraint_enforcer(samples, rng, L)

    def plot_corner_diagnostics(
        self,
        stock_prices: np.ndarray,
        parameter_samples: Dict[str, np.ndarray],
        current_stock_price: Optional[float] = None,
        quantiles: list = [0.16, 0.5, 0.84],
        show_titles: bool = True,
        figsize: Tuple[float, float] = (12, 8),
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create corner diagnostic plots of all parameters including stock_price.

        Parameters:
        -----------
        stock_prices : np.ndarray
            Array of simulated stock prices
        parameter_samples : Dict[str, np.ndarray]
            Dictionary of parameter samples from simulation
        current_stock_price : float, optional
            If provided, draws a red line on all subplots that have stock_price as one dimension
        quantiles : list
            Quantiles to show on the plots (default: [0.16, 0.5, 0.84])
        show_titles : bool
            Whether to show titles on diagonal plots (default: True)
        figsize : tuple
            Figure size (default: (12, 8))
        **kwargs : additional arguments
            Additional arguments passed to corner.corner()

        Returns:
        --------
        tuple : (fig, axes) matplotlib figure and axes objects
        """
        logger.info("Creating diagnostic corner plots")

        # Prepare data with stock_price as first column
        data_dict = {"stock_price": stock_prices}
        data_dict.update(parameter_samples)

        # Convert to array with stock_price first
        samples = np.column_stack([data_dict[key] for key in data_dict.keys()])

        # Create labels with proper formatting
        labels = []
        for key in data_dict.keys():
            if key == "stock_price":
                labels.append(r"Stock Price (\$)")  # Escape dollar sign for LaTeX
            elif key.endswith("_rate") or key.endswith("_growth"):
                labels.append(key.replace("_", " ").title() + " (%)")
            elif key.endswith("_horizon"):
                labels.append(key.replace("_", " ").title() + " (years)")
            else:
                labels.append(key.replace("_", " ").title())

        # Create corner plot
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=quantiles,
            show_titles=show_titles,
            title_kwargs={"fontsize": 12},
            fig_size=figsize,
            **kwargs,
        )

        # Add current stock price lines if provided
        if current_stock_price is not None:
            axes = fig.get_axes()
            n_params = len(labels)

            # Add vertical line to plots in first column (stock_price on x-axis)
            for i in range(n_params):
                ax = axes[i * n_params + 0]
                ax.axvline(current_stock_price, color="red", linestyle="--", linewidth=2, alpha=0.8)

        plt.tight_layout()
        logger.info("Corner plots generated successfully")

        return fig, fig.get_axes()

    def plot_terms_diagnostics(
        self,
        stock_prices: np.ndarray,
        pv_fcf_fractions: np.ndarray,
        figsize: Tuple[float, float] = (8, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create diagnostic plot showing FCF vs Terminal value contributions to stock price.

        This plot visualizes the relationship between the FCF term and Terminal term
        components of the DCF valuation, with density contours and diagonal lines
        representing total stock price levels.

        Parameters:
        -----------
        stock_prices : np.ndarray
            Array of simulated stock prices
        pv_fcf_fractions : np.ndarray
            Array of present value FCF fractions
        figsize : tuple
            Figure size (default: (8, 6))

        Returns:
        --------
        tuple : (fig, ax) matplotlib figure and axes objects
        """
        logger.info("Creating DCF terms diagnostic plot")

        # Sample subset for performance
        n_subset = min(10000, len(stock_prices))
        subset_idx = np.random.choice(len(stock_prices), size=n_subset, replace=False)
        stock_prices_subset = stock_prices[subset_idx]
        pv_fractions = pv_fcf_fractions[subset_idx]

        # Calculate FCF and Terminal components
        x = pv_fractions * stock_prices_subset
        y = (1 - pv_fractions) * stock_prices_subset

        # Filter outliers using percentiles
        x_min, x_max = np.percentile(x, 0.1), np.percentile(x, 99.9)
        y_min, y_max = np.percentile(y, 0.1), np.percentile(y, 99.9)
        valid_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x, y, stock_prices_subset = x[valid_mask], y[valid_mask], stock_prices_subset[valid_mask]

        # Create KDE and evaluate on grid
        kde = gaussian_kde(np.vstack([x, y]))
        densities = kde(np.vstack([x, y]))

        n_grid = 200
        xi, yi = np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = np.reshape(kde(np.vstack([Xi.ravel(), Yi.ravel()])).T, Xi.shape)

        # Setup plot and scatter
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(x, y, c=stock_prices_subset, cmap="viridis", s=10, alpha=0.5, edgecolors="none")

        # Add density contours
        percentiles = [25, 50, 75, 90]
        levels = np.percentile(densities, percentiles)
        ax.contour(Xi, Yi, Zi, levels=levels, colors="white", linewidths=3.0, alpha=0.6)
        contour_colored = ax.contour(Xi, Yi, Zi, levels=levels, cmap="viridis", linewidths=1.5)

        # Label contours with white halo effect
        fmt = {level: f"{pct}th" for level, pct in zip(levels, percentiles, strict=True)}
        texts = ax.clabel(contour_colored, levels, fmt=fmt, inline=True, fontsize=12, colors="black")
        for txt in texts:
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        # Add diagonal stock price lines
        price_levels = np.linspace(np.percentile(stock_prices_subset, 5), np.percentile(stock_prices_subset, 95), 4)
        norm = Normalize(vmin=stock_prices_subset.min(), vmax=stock_prices_subset.max())
        cmap = plt.get_cmap("viridis")
        y_margin, x_margin = 0.02 * (y_max - y_min), 0.075 * (x_max - x_min)

        for price_val in price_levels:
            x_line = np.linspace(x_min, x_max, 500)
            y_line = price_val - x_line
            valid = (y_line >= y_min) & (y_line <= y_max)

            if np.any(valid):
                x_plot, y_plot = x_line[valid], y_line[valid]
                color = cmap(norm(price_val))
                ax.plot(x_plot, y_plot, color="white", linewidth=3, alpha=0.5)
                ax.plot(x_plot, y_plot, color=color, linewidth=2.0)

                # Position and adjust label location
                x_label = x_min + 0.95 * (x_max - x_min)
                y_label = price_val - x_label

                if y_label < y_min + y_margin:
                    y_label = y_min + y_margin
                    x_label = price_val - y_label
                elif y_label > y_max - y_margin:
                    y_label = y_max - y_margin
                    x_label = price_val - y_label

                if x_label < x_min + x_margin:
                    x_label = x_min + x_margin
                    y_label = price_val - x_label
                elif x_label > x_max - x_margin:
                    x_label = x_max - x_margin
                    y_label = price_val - x_label

                ax.text(
                    x_label,
                    y_label,
                    f"${price_val:.0f}",
                    color=color,
                    va="center",
                    ha="left",
                    fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.5),
                )

        # Add y=x reference line
        max_axis = max(abs(x).max(), abs(y).max())
        ax.plot([0, max_axis], [0, max_axis], "k-", linewidth=1.0, alpha=0.7)
        ax.text(x_max, x_max, "y = x", color="black", fontsize=12, ha="right", va="bottom", alpha=0.8)

        # Formatting
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Stock Price (FCF Term)")
        ax.set_ylabel("Stock Price (Terminal Term)")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Total Stock Price")

        plt.tight_layout()
        logger.info("DCF terms diagnostic plot generated successfully")

        return fig, ax


class StandardDCFModel(BaseDCFModel):
    """
    Standard DCF model implementation with configurable parameters.

    Required parameters for calculate_dcf:
    - discount_rate: Required rate of return (float)
    - growth_rate: Revenue/FCF growth rate (float, optional - can be calculated from data)
    - terminal_growth: Perpetual growth rate (float)
    - cagr_lookback: Number of years for historical CAGR calculation (float)

    Usage:
    ------
    model = StandardDCFModel()
    model.configure_priors(distributions, correlation_matrix)
    
    # Returns values directly instead of storing as attributes
    pv_fcf_fractions, dcf_values, stock_prices, parameter_samples = model.simulate(df, n_samples=1000)
    
    # Or use sliding window analysis
    results_df = model.sliding_dcf_analysis(df, cagr_lookback=5.0, forecast_horizon=10, n_samples=1000)
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize StandardDCFModel with financial data."""
        super().__init__(df)

    def configure_priors(self, distributions: Dict[str, Any], correlation_matrix: Optional[np.ndarray] = None) -> None:
        """
        Configure prior distributions and correlation matrix for the model with enhanced validation.

        Parameters:
        -----------
        distributions : dict
            Dictionary mapping parameter names to scipy.stats distributions.
            Required keys: ['discount_rate_scale', 'terminal_growth', 'cagr_lookback']
            Optional keys: ['growth_rate'] - if None, will use empirical growth rate with correlations preserved
        correlation_matrix : np.ndarray, optional
            Correlation matrix for parameters. If None, uses identity matrix (independent sampling).
            If shape doesn't match distributions, it will be adjusted automatically.


        Special handling for growth_rate:
        - If None: Will use empirical CAGR calculated from historical data, preserving correlations
        - If numeric (int/float): Converted to delta function distribution using norm(loc=value, scale=value*1e-6)
        - If scipy.stats distribution: Used as-is
        
        Example:
        --------
        from scipy.stats import uniform, norm

        distributions = {
            'discount_rate_scale': uniform(loc=0.05, scale=0.15),
            'growth_rate': None,  # Will be calculated from historical data with correlations preserved
            'terminal_growth': uniform(loc=0.01, scale=0.04),
            'cagr_lookback': uniform(loc=5, scale=5)
        }

        # Correlation matrix for all 4 parameters - correlations will be preserved even for empirical growth_rate
        correlation = np.array([
            [1.0, 0.8, 0.2, 0.5],
            [0.8, 1.0, 0.3, 0.6],
            [0.2, 0.3, 1.0, 0.4],
            [0.5, 0.6, 0.4, 1.0]
        ])

        model.configure_priors(distributions, correlation)
        """
        logger.debug("Configuring prior distributions")

        # Store original inputs for set_center method
        self._original_distributions = distributions
        self._original_correlation_matrix = correlation_matrix

        # Required parameters (growth_rate is optional)
        required_params = ["discount_rate_scale", "terminal_growth", "cagr_lookback"]
        missing_params = [p for p in required_params if p not in distributions]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Process distributions - handle special case for growth_rate
        processed_distributions = {}
        param_types = {}
        empirical_growth_rate = None
        
        for param_name, dist in distributions.items():
            if param_name == "growth_rate":
                if dist is None:
                    # Calculate empirical growth rate and create distribution around it
                    # Calculate empirical growth rate using default horizon for initial estimate
                    annual_fcf = self.df["FCF-LTM"]
                    last_fcf = annual_fcf.iloc[-1]
                    
                    # FCF-LTM is rolling LTM for each quarter. 10 years is 10 * 4 quarters
                    cagr_lookback_years = 10.0
                    offset = int(cagr_lookback_years * 4 + 1)
                    if len(annual_fcf) >= offset:
                        fcf_from_n_years_ago = annual_fcf.iloc[-offset]
                        empirical_growth_rate = (last_fcf / fcf_from_n_years_ago) ** (1 / cagr_lookback_years) - 1
                    else:
                        empirical_growth_rate = annual_fcf.pct_change().mean()
                    
                    # Create a narrow distribution centered on empirical value to preserve correlations
                    # Use small scale to approximate delta function while allowing correlation sampling
                    scale = abs(empirical_growth_rate) * 0.01 if empirical_growth_rate != 0 else 0.001
                    processed_distributions[param_name] = stats.norm(loc=empirical_growth_rate, scale=scale)
                    param_types[param_name] = "empirical_with_correlation"
                    logger.debug(f"growth_rate: Using empirical CAGR {empirical_growth_rate:.2%} with correlations preserved")
                elif isinstance(dist, int | float):
                    # Convert numeric to delta function
                    scale = abs(dist) * 1e-6 if dist != 0 else 1e-6
                    processed_distributions[param_name] = stats.norm(loc=dist, scale=scale)
                    param_types[param_name] = "delta_function"
                    logger.debug(f"growth_rate: Converted {dist} to delta function distribution")
                elif hasattr(dist, 'ppf'):
                    # Valid scipy.stats distribution
                    processed_distributions[param_name] = dist
                    param_types[param_name] = "distribution"
                    logger.debug(f"growth_rate: Using provided distribution")
                else:
                    raise ValueError(f"growth_rate must be None, numeric, or scipy.stats distribution, got {type(dist)}")
            elif hasattr(dist, 'ppf'):
                processed_distributions[param_name] = dist
                param_types[param_name] = "distribution"
                logger.debug(f"{param_name}: Using provided distribution")
            elif isinstance(dist, int | float):
                # Convert numeric to delta function
                scale = abs(dist) * 1e-6 if dist != 0 else 1e-6
                processed_distributions[param_name] = stats.norm(loc=dist, scale=scale)
                param_types[param_name] = "delta_function"
                logger.debug(f"{param_name}: Converted {dist} to delta function distribution")
            else:
                raise ValueError(f"{param_name} must be numeric or scipy.stats distribution, got {type(dist)}")

        # Store empirical growth rate for use in simulation
        self._empirical_growth_rate = empirical_growth_rate

        # Set up parameter names (include growth_rate even if empirical for correlation matrix consistency)
        all_param_names = ["discount_rate_scale", "growth_rate", "terminal_growth", "cagr_lookback"]
        # Filter to only parameters that are provided
        self.parameter_names = [p for p in all_param_names if p in distributions]
        
        # Store distributions and types
        self.distributions = processed_distributions
        self._param_types = param_types
        
        # Handle correlation matrix
        if correlation_matrix is not None:
            # Check if correlation matrix needs adjustment
            n_provided_params = len(self.parameter_names)
            
            if correlation_matrix.shape != (n_provided_params, n_provided_params):
                logger.warning(f"Correlation matrix shape {correlation_matrix.shape} doesn't match " +
                            f"provided parameters {n_provided_params}. Adjusting correlation matrix...")
                
                # Extract relevant submatrix based on provided parameters
                param_indices = [all_param_names.index(p) for p in self.parameter_names]
                adjusted_corr = correlation_matrix[np.ix_(param_indices, param_indices)]
                
                self._correlation_matrix = adjusted_corr
                logger.debug(f"Adjusted correlation matrix to shape {adjusted_corr.shape}")
            else:
                self._correlation_matrix = correlation_matrix
                logger.debug(f"Using provided correlation matrix with shape {correlation_matrix.shape}")
            
            # Validate correlation matrix properties
            if not np.allclose(self._correlation_matrix, self._correlation_matrix.T):
                logger.warning("Correlation matrix is not symmetric. Symmetrizing...")
                self._correlation_matrix = (self._correlation_matrix + self._correlation_matrix.T) / 2
            
            eigenvals = np.linalg.eigvals(self._correlation_matrix)
            if np.any(eigenvals < -1e-8):
                logger.warning("Correlation matrix may not be positive semidefinite")
        else:
            self._correlation_matrix = None
            logger.debug("No correlation matrix provided - will use independent sampling")

        logger.debug("Configuration complete")
        logger.debug(f"Parameters: {self.parameter_names}")
        logger.debug(f"Distributions: {len(self.distributions)}")
        logger.debug(f"Empirical w/ correlation: {[k for k, v in param_types.items() if v == 'empirical_with_correlation']}")
        logger.debug(f"Delta functions: {[k for k, v in param_types.items() if v == 'delta_function']}")
        logger.debug(f"Standard distributions: {[k for k, v in param_types.items() if v == 'distribution']}")
        logger.debug(f"Correlation matrix: {'Provided' if self._correlation_matrix is not None else 'None (independent)'}")

    def calculate_dcf(
        self,
        discount_rate_scale: float = 1.0,
        growth_rate: Optional[float] = None,
        terminal_growth: float = 0.025,
        cagr_lookback: float = 10,
        forecast_horizon: int = 10,
    ) -> Tuple[float, float, float]:
        """
        Calculate DCF value using standard methodology with WACC from dataframe.

        Parameters:
        -----------
        discount_rate_scale : float
            Scaling factor applied to WACC to get discount rate (default=1.0)
            discount_rate = WACC * discount_rate_scale
        growth_rate : float, optional
            If provided, use this growth rate instead of calculating from historical FCF
        terminal_growth : float
            Perpetual growth rate after forecast horizon (as decimal)
        cagr_lookback : float
            Number of years looking back for CAGR calculation (default=10)
        forecast_horizon : int
            Number of years to forecast into future before terminal value plateau

        Returns:
        --------
        tuple : (pv_fcf_fraction, dcf_value, stock_price)
        """
        # Prepare data
        annual_fcf = self.df["FCF-LTM"]
        shares_outstanding = self.df["Shares-Outstanding"].iloc[-1]
        last_fcf = annual_fcf.iloc[-1]  # Most recent year's FCF

        # Get WACC and apply scaling factor
        wacc = self.df["WACC"].iloc[-1]  # Use most recent WACC
        discount_rate = wacc * discount_rate_scale

        # Determine CAGR (g)
        if growth_rate is not None:
            g = growth_rate
        else:
            # FCF-LTM is rolling LTM for each quarter. cagr_lookback years is cagr_lookback * 4 quarters
            offset = int(cagr_lookback * 4 + 1)
            if len(annual_fcf) >= offset:
                fcf_from_n_years_ago = annual_fcf.iloc[-offset]
                g = (last_fcf / fcf_from_n_years_ago) ** (1 / cagr_lookback) - 1
            else:
                g = annual_fcf.pct_change().mean()

        # Project FCF for Explicit Horizon
        projected = [last_fcf * (1 + g) ** t for t in range(1, forecast_horizon + 1)]

        # Compute Terminal Value
        terminal = projected[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # Discount Projected FCFs and Terminal Value
        pv_fcf = sum(fc / (1 + discount_rate) ** t for t, fc in enumerate(projected, start=1))
        pv_term = terminal / (1 + discount_rate) ** forecast_horizon
        dcf_val = pv_fcf + pv_term

        pv_fcf_fraction = pv_fcf / (pv_fcf + pv_term)

        # Implied Stock Price
        stock_price = dcf_val / shares_outstanding

        return pv_fcf_fraction, dcf_val, stock_price

    def create_summary_table(self, **params) -> Table:
        """Create detailed Rich table for standard DCF model results."""
        table = Table(show_header=False, box=box.SIMPLE, pad_edge=True)
        table.add_column("Metric", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Format parameters nicely
        if "discount_rate_scale" in params:
            table.add_row("WACC Scale Factor", f"{params['discount_rate_scale']:.2f}x")
        if "terminal_growth" in params:
            table.add_row("Perpetual Growth (g∞)", f"{params['terminal_growth']:.2%}")
        if "forecast_horizon" in params:
            table.add_row("Explicit Horizon (T)", f"{params['forecast_horizon']} years")
        if "growth_rate" in params:
            table.add_row("Historical/Used CAGR (g)", f"{params['growth_rate']:.2%}")

        # Add financial results if provided
        if "wacc" in params:
            table.add_row("Base WACC", f"{params['wacc']:.2%}")
        if "discount_rate" in params:
            table.add_row("Effective Discount Rate", f"{params['discount_rate']:.2%}")
        for key in ["last_fcf", "pv_fcf", "pv_terminal", "dcf_value", "shares_outstanding", "stock_price"]:
            if key in params:
                value = params[key]
                if key == "stock_price":
                    table.add_row("Implied Stock Price", f"${value:,.2f}")
                elif key == "shares_outstanding":
                    table.add_row("Shares Outstanding", f"{value:,.0f}")
                elif "fcf" in key or "pv" in key or "dcf" in key:
                    table.add_row(key.replace("_", " ").title(), f"${value:,.0f}M")
                else:
                    table.add_row(key.replace("_", " ").title(), f"{value}")

        return table
