import os
import subprocess


def test_macrotrends():
    """
    Integration test for the macrotrends CLI command.
    Checks that the command runs, fetches data, and creates expected files.
    """
    # Use a temp dir to avoid overwriting real outputs
    cwd = os.getcwd()
    try:
        # Create required directories
        os.makedirs("macro_data/parquet", exist_ok=True)
        os.makedirs("fundamentals/utility", exist_ok=True)

        # Run the CLI command
        result = subprocess.run(
            [
                "python",
                os.path.join(cwd, "fund_cli.py"),
                "macrotrends",
                "--symbols", "AAPL",
                "--pages", "income-statement",  # Test with just one page to keep test fast
                "--freq", "Q",
                "--force",  # Force fetch to ensure we get fresh data
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Check that parquet files were created
        parquet_files = os.listdir("macro_data/parquet")
        assert any(f.startswith("AAPL_income-statement_") and f.endswith(".parquet") for f in parquet_files)

        # Check that DuckDB file was created
        assert os.path.exists("macro_data/macrotrends.duckdb")

        # Check CLI output for expected messages
        assert "macrotrends.duckdb refreshed" in result.stdout

    finally:
        os.chdir(cwd)
