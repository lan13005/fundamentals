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
                "--symbols",
                "AAPL",
                "--freq",
                "Q",
                "--force",  # Force fetch to ensure we get fresh data
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Check that merged parquet file was created
        parquet_files = os.listdir("macro_data/parquet")
        # Find the expected merged file name (AAPL_<quarter>.parquet)
        merged_file = None
        for f in parquet_files:
            if f.startswith("AAPL_") and f.endswith(".parquet") and f.count("_") == 1:
                merged_file = f
                break
        assert merged_file is not None, f"Merged parquet file not found in {parquet_files}"

        # Check CLI output for expected messages
        assert "✓ Loaded" in result.stdout

    finally:
        os.chdir(cwd)
