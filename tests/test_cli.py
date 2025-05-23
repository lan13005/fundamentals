import os
import subprocess


def test_sector_industry_list_cli(tmp_path):
    """
    Integration test for the sector-industry-list CLI command.
    Checks that the command runs, produces the expected CSVs, and prints summary stats.
    """
    # Use a temp dir to avoid overwriting real outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = subprocess.run(
            [
                "python",
                os.path.join(cwd, "fund_cli.py"),
                "sector-industry-list",
                "--market-category",
                "Q",
                "--max-tickers",
                "10",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check output files
        assert os.path.exists("yahoo_sector_industry_summary.csv")
        assert os.path.exists("yahoo_company_info.csv")
        # Check output content
        with open("yahoo_sector_industry_summary.csv") as f:
            content = f.read()
            assert "Total Market Cap" in content
        # Check CLI output for stats
        assert "Loaded" in result.stdout
        assert "Sector-Industry Summary" in result.stdout
    finally:
        os.chdir(cwd)
