"""
Global logging configuration for the fundamentals package.

This module provides centralized logging configuration that can be controlled
via environment variables stored in a .env file.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.logging import RichHandler

# Load environment variables
load_dotenv()

console = Console()

# Valid logging levels
VALID_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default logging level
DEFAULT_LOG_LEVEL = "INFO"

# Global logger cache to avoid reconfiguring multiple times
_configured_loggers = set()


def get_project_root() -> Path:
    """Get the project root directory."""
    # From fundamentals/utility/logging_config.py, go up to project root
    current_file = Path(__file__)
    # Go up: logging_config.py -> utility -> fundamentals (package) -> project_root
    return current_file.parent.parent.parent


def get_env_file_path() -> Path:
    """Get the path to the .env file."""
    return get_project_root() / ".env"


def get_logging_level() -> str:
    """
    Get the current logging level from environment variables.
    
    Returns:
        str: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = os.getenv("FUNDAMENTALS_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return level if level in VALID_LEVELS else DEFAULT_LOG_LEVEL


def set_logging_level(level: str) -> None:
    """
    Set the global logging level and save it to the .env file.
    
    Parameters:
    -----------
    level : str
        Logging level - "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        
    Raises:
    -------
    ValueError: If the level is not valid
    """
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {VALID_LEVELS}")
    
    env_file = get_env_file_path()
    
    # Create .env file if it doesn't exist
    if not env_file.exists():
        env_file.touch()
    
    # Set the logging level in the .env file
    set_key(str(env_file), "FUNDAMENTALS_LOG_LEVEL", level)
    
    # Update the current environment
    os.environ["FUNDAMENTALS_LOG_LEVEL"] = level
    
    # Reconfigure all existing loggers
    _reconfigure_all_loggers()
    
    console.print(f"[green]Global logging level set to {level}[/green]")


def configure_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Configure a logger with rich formatting and the global logging level.
    
    Parameters:
    -----------
    name : str
        Logger name (typically __name__ or module name)
    level : str, optional
        Override the global logging level for this specific logger
        
    Returns:
    --------
    logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if name not in _configured_loggers:
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Set up rich handler
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            console=console
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger.addHandler(handler)
        logger.propagate = False
        
        _configured_loggers.add(name)
    
    # Set the logging level
    log_level = level or get_logging_level()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)
    
    return logger


def _reconfigure_all_loggers() -> None:
    """Reconfigure all previously configured loggers with the new level."""
    new_level = get_logging_level()
    numeric_level = getattr(logging, new_level, logging.INFO)
    
    for logger_name in _configured_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    This is the main function that modules should use to get their logger.
    
    Parameters:
    -----------
    name : str
        Logger name (typically __name__)
        
    Returns:
    --------
    logging.Logger: Configured logger instance
    """
    return configure_logger(name)


# Initialize the logging system
def init_logging() -> None:
    """Initialize the global logging system."""
    # Configure root logger to prevent duplicate messages
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Set a minimal handler to capture any unconfigured loggers
        handler = RichHandler(console=console, show_time=False, show_path=False)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)  # Only show warnings and errors from unconfigured loggers


# Initialize on import
init_logging()
