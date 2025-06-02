# fundamentals

Python package for fundamental analysis of companies for value / quality investing:
- use LLMs to study current macro-trends to identify tailwinds and headwinds
- perform company-specific analyses
  - SEC filings (10-K, 10-Q, etc.)
  - financial analysis (income statement, balance sheet, cash flow statement, financial ratios)

**Features:**
- aimed for Cursor IDE with rules- and task-based agentic programming
- using language models to generate reports and distil textual information using MCP tools

# Installation
Using [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```
