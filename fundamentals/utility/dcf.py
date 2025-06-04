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
from scipy.stats import gaussian_kde, norm

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

    def __init__(self):
        """Initialize the DCF model."""
        self.distributions = self._define_prior_distributions()
        self.parameter_names = list(self.distributions.keys())
        self._correlation_matrix = None

        # Instance variables to store simulation results
        self.dcf_values: Optional[np.ndarray] = None
        self.stock_prices: Optional[np.ndarray] = None
        self.parameter_samples: Optional[Dict[str, np.ndarray]] = None

    @abstractmethod
    def _define_prior_distributions(self) -> Dict[str, Any]:
        """
        Define prior distributions for model parameters.

        Returns:
        --------
        dict : Dictionary mapping parameter names to scipy.stats distributions
        """
        pass

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

    @abstractmethod
    def calculate_dcf(self, df: pd.DataFrame, **params) -> Tuple[float, float, float]:
        """
        Calculate DCF value and implied stock price for given parameters.

        Parameters:
        -----------
        df : pd.DataFrame
            Financial data with required columns including 'WACC'
        **params : keyword arguments
            Model parameters (discount_rate_scale, growth_rate, etc.)

        Returns:
        --------
        tuple : (pv_fcf_fraction, dcf_value, stock_price)
        """
        pass

    def simulate(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        random_state: int = 42,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run Monte Carlo simulation for DCF valuation with parameter sampling.
        Stores results in instance variables: dcf_values, stock_prices, parameter_samples.

        Parameters:
        -----------
        df : pd.DataFrame
            Financial data
        n_samples : int
            Number of samples to generate
        random_state : int
            Random seed for reproducibility
        correlation_matrix : np.ndarray, optional
            Correlation matrix for parameters. If None, uses _get_correlation_matrix()
        """
        if not hasattr(self, "distributions") or not self.distributions:
            raise ValueError("Prior distributions not configured. Call configure_priors() first.")

        console.rule("[bold blue]Monte Carlo DCF Simulation")

        # Sample parameters
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            # Task 1: Sample parameters
            sample_task = progress.add_task("Sampling parameters...", total=100)

            rng = np.random.default_rng(random_state)

            # Get correlation matrix
            if correlation_matrix is None:
                corr = self._get_correlation_matrix()
            else:
                corr = correlation_matrix

            n_params = len(self.parameter_names)

            if corr.shape != (n_params, n_params):
                raise ValueError(f"Correlation matrix shape {corr.shape} doesn't match number of parameters {n_params}")

            # Gaussian copula
            L = np.linalg.cholesky(corr)
            Z = rng.standard_normal(size=(n_samples, n_params))
            correlated_Z = Z @ L.T
            U = norm.cdf(correlated_Z)  # uniform samples with preserved rank correlation

            progress.update(sample_task, advance=50)

            # Transform to marginal distributions
            samples = {}
            for i, param_name in enumerate(self.parameter_names):
                distribution = self.distributions[param_name]
                samples[param_name] = distribution.ppf(U[:, i])

            # Apply constraints with rejection sampling
            samples = self._apply_parameter_constraints(samples, rng, L)
            self.parameter_samples = samples

            progress.update(sample_task, advance=50)

            # Task 2: Run DCF simulations
            sim_task = progress.add_task("Running DCF simulations...", total=n_samples)

            # Initialize arrays to store results
            pv_fcf_fractions = np.zeros(n_samples)
            dcf_values = np.zeros(n_samples)
            stock_prices = np.zeros(n_samples)

            for i in range(n_samples):
                # Extract parameters for this iteration
                params = {name: samples[name][i] for name in self.parameter_names}

                # Calculate DCF for this parameter set
                pv_fcf_fraction, dcf_val, stock_price = self.calculate_dcf(df, **params)
                pv_fcf_fractions[i] = pv_fcf_fraction
                dcf_values[i] = dcf_val
                stock_prices[i] = stock_price

                if i % max(1, n_samples // 100) == 0:  # Update every 1%
                    progress.update(sim_task, advance=max(1, n_samples // 100))

            # Store results
            self.pv_fcf_fractions = pv_fcf_fractions
            self.dcf_values = dcf_values
            self.stock_prices = stock_prices

            # Print summary
            console.print(
                f"PV FCF Fractions: {pv_fcf_fractions.mean():.2f} ± {pv_fcf_fractions.std():.2f}", style="green"
            )
            console.print(f"DCF values: {dcf_values.mean():.2f} ± {dcf_values.std():.2f}", style="green")
            console.print(f"Stock prices: {stock_prices.mean():.2f} ± {stock_prices.std():.2f}", style="green")

        console.print("Access the values as:", style="bold green")
        console.print(f" - self.pv_fcf_fractions (shape): {self.pv_fcf_fractions.shape}", style="green")
        console.print(f" - self.dcf_values (shape): {self.dcf_values.shape}", style="green")
        console.print(f" - self.stock_prices (shape): {self.stock_prices.shape}", style="green")
        console.print(f" - self.parameter_samples (Dict keys): {list(self.parameter_samples.keys())}", style="green")
        console.print(f"[bold green]✓[/bold green] Simulation complete: {n_samples:,} samples generated", style="green")

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

        # For safety, ensure terminal growth is less than discount_rate_scale
        # This is a conservative constraint assuming WACC > 0
        valid_mask = samples["terminal_growth"] < samples["discount_rate_scale"]
        while not np.all(valid_mask):
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

            valid_mask = samples["terminal_growth"] < samples["discount_rate_scale"]

        return samples

    def create_summary_table(self, **params) -> Table:
        """
        Create Rich table summarizing model parameters.
        Override in subclasses for model-specific formatting.
        """
        table = Table(show_header=False, box=box.SIMPLE, pad_edge=True)
        table.add_column("Parameter", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        for param_name, value in params.items():
            if param_name.endswith("_rate") or param_name.endswith("_growth"):
                table.add_row(param_name.replace("_", " ").title(), f"{value:.2%}")
            elif param_name.endswith("_horizon"):
                table.add_row(param_name.replace("_", " ").title(), f"{value:.1f} years")
            else:
                table.add_row(param_name.replace("_", " ").title(), f"{value}")

        return table

    def plot_corner_diagnostics(
        self,
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
        if self.parameter_samples is None or self.stock_prices is None:
            raise ValueError("Simulation results not available. Run simulate() first.")

        console.print("[bold blue]Creating diagnostic corner plots...", style="blue")

        # Prepare data with stock_price as first column
        data_dict = {"stock_price": self.stock_prices}
        data_dict.update(self.parameter_samples)

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
        console.print("[bold green]✓[/bold green] Corner plots generated successfully", style="green")

        return fig, fig.get_axes()

    def plot_terms_diagnostics(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create diagnostic plot showing FCF vs Terminal value contributions to stock price.

        This plot visualizes the relationship between the FCF term and Terminal term
        components of the DCF valuation, with density contours and diagonal lines
        representing total stock price levels.

        Returns:
        --------
        tuple : (fig, ax) matplotlib figure and axes objects
        """
        if self.parameter_samples is None or self.stock_prices is None:
            raise ValueError("Simulation results not available. Run simulate() first.")

        console.print("[bold blue]Creating DCF terms diagnostic plot...", style="blue")

        # Sample subset for performance
        n_subset = min(10000, len(self.stock_prices))
        subset_idx = np.random.choice(len(self.stock_prices), size=n_subset, replace=False)
        stock_prices = self.stock_prices[subset_idx]
        pv_fractions = self.pv_fcf_fractions[subset_idx]

        # Calculate FCF and Terminal components
        x = pv_fractions * stock_prices
        y = (1 - pv_fractions) * stock_prices

        # Filter outliers using percentiles
        x_min, x_max = np.percentile(x, 0.1), np.percentile(x, 99.9)
        y_min, y_max = np.percentile(y, 0.1), np.percentile(y, 99.9)
        valid_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x, y, stock_prices = x[valid_mask], y[valid_mask], stock_prices[valid_mask]

        # Create KDE and evaluate on grid
        kde = gaussian_kde(np.vstack([x, y]))
        densities = kde(np.vstack([x, y]))

        n_grid = 200
        xi, yi = np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = np.reshape(kde(np.vstack([Xi.ravel(), Yi.ravel()])).T, Xi.shape)

        # Setup plot and scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=stock_prices, cmap="viridis", s=10, alpha=0.5, edgecolors="none")

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
        price_levels = np.linspace(np.percentile(stock_prices, 5), np.percentile(stock_prices, 95), 4)
        norm = Normalize(vmin=stock_prices.min(), vmax=stock_prices.max())
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
        console.print("[bold green]✓[/bold green] DCF terms diagnostic plot generated successfully", style="green")

        return fig, ax


class StandardDCFModel(BaseDCFModel):
    """
    Standard DCF model implementation with configurable parameters.

    Required parameters for calculate_dcf:
    - discount_rate: Required rate of return (float)
    - growth_rate: Revenue/FCF growth rate (float, optional - can be calculated from data)
    - terminal_growth: Perpetual growth rate (float)
    - time_horizon: Number of years for historical CAGR calculation (float)

    Usage:
    ------
    model = StandardDCFModel()
    model.configure_priors(distributions, correlation_matrix)
    model.simulate(df, n_samples=1000)
    # Results available in: model.dcf_values, model.stock_prices, model.parameter_samples
    """

    def __init__(self):
        """Initialize StandardDCFModel without predefined distributions."""
        self.distributions = {}
        self.parameter_names = []
        self._correlation_matrix = None

        # Instance variables to store simulation results
        self.dcf_values: Optional[np.ndarray] = None
        self.stock_prices: Optional[np.ndarray] = None
        self.parameter_samples: Optional[Dict[str, np.ndarray]] = None

    def _define_prior_distributions(self) -> Dict[str, Any]:
        """Not implemented - use configure_priors() instead."""
        return {}

    def configure_priors(self, distributions: Dict[str, Any], correlation_matrix: Optional[np.ndarray] = None) -> None:
        """
        Configure prior distributions and correlation matrix for the model.

        Parameters:
        -----------
        distributions : dict
            Dictionary mapping parameter names to scipy.stats distributions.
            Required keys: ['discount_rate_scale', 'growth_rate', 'terminal_growth', 'time_horizon']
        correlation_matrix : np.ndarray, optional
            Correlation matrix for parameters. If None, uses identity matrix (independent sampling).

        Example:
        --------
        from scipy.stats import uniform, norm

        distributions = {
            'discount_rate_scale': uniform(loc=0.05, scale=0.15),
            'growth_rate': norm(loc=0.05, scale=0.10),
            'terminal_growth': uniform(loc=0.01, scale=0.04),
            'time_horizon': uniform(loc=5, scale=5)
        }

        correlation = np.array([
            [1.0, 0.8, 0.2, 0.5],
            [0.8, 1.0, 0.3, 0.6],
            [0.2, 0.3, 1.0, 0.4],
            [0.5, 0.6, 0.4, 1.0]
        ])

        model.configure_priors(distributions, correlation)
        """
        required_params = ["discount_rate_scale", "growth_rate", "terminal_growth", "time_horizon"]
        missing_params = [p for p in required_params if p not in distributions]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        self.distributions = distributions
        self.parameter_names = list(distributions.keys())
        self._correlation_matrix = correlation_matrix

        if correlation_matrix is not None:
            n_params = len(self.parameter_names)
            if correlation_matrix.shape != (n_params, n_params):
                raise ValueError(
                    f"Correlation matrix shape {correlation_matrix.shape} doesn't match number of parameters {n_params}"
                )

    def calculate_dcf(
        self,
        df: pd.DataFrame,
        discount_rate_scale: float = 1.0,
        growth_rate: Optional[float] = None,
        terminal_growth: float = 0.025,
        time_horizon: float = 10,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate DCF value using standard methodology with WACC from dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'FCF-LTM', 'Shares-Outstanding', and 'WACC' columns
        discount_rate_scale : float
            Scaling factor applied to WACC to get discount rate (default=1.0)
            discount_rate = WACC * discount_rate_scale
        growth_rate : float, optional
            If provided, use this growth rate instead of calculating from historical FCF
        terminal_growth : float
            Perpetual growth rate after forecast horizon (as decimal)
        time_horizon : float
            Number of years to look back for CAGR calculation (default=10)
        forecast_horizon : int, optional
            Number of years in explicit forecast (defaults to int(time_horizon))

        Returns:
        --------
        tuple : (pv_fcf_fraction, dcf_value, stock_price)
        """
        if forecast_horizon is None:
            forecast_horizon = int(time_horizon)

        # Prepare data
        annual_fcf = df["FCF-LTM"]
        shares_outstanding = df["Shares-Outstanding"].iloc[-1]
        last_fcf = annual_fcf.iloc[-1]  # Most recent year's FCF
        
        # Get WACC and apply scaling factor
        wacc = df["WACC"].iloc[-1]  # Use most recent WACC
        discount_rate = wacc * discount_rate_scale

        # Determine CAGR (g)
        if growth_rate is not None:
            g = growth_rate
        else:
            # FCF-LTM is rolling LTM for each quarter. time_horizon years is time_horizon * 4 quarters
            offset = int(time_horizon * 4 + 1)
            if len(annual_fcf) >= offset:
                fcf_from_n_years_ago = annual_fcf.iloc[-offset]
                g = (last_fcf / fcf_from_n_years_ago) ** (1 / time_horizon) - 1
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
