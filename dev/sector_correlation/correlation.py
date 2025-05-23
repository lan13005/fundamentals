# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import numpy as np
import yfinance as yf

# ---- user parameters -------------------------------------------------------
available_tickers = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Capital Goods": "IYJ",
    "Comm & Prof Svcs": "XLI",
    "Transportation": "IYT",
    "Autos & Components": "CARZ",
    "Cons Durables & Apparel": "IBUY",
    "Consumer Services": "PEJ",
    "Retailing": "XRT",
    "Food & Staples Retail": "XLP",
    "Food/Beverage/Tobacco": "PBJ",
    "Household & Pers Prod": "IYK",
    "HC Equip & Svcs": "IHI",
    "Pharma / Biotech": "XPH",
    "Banks": "KBE",
    "Diversified Financials": "IYF",
    "Insurance": "KIE",
    "Real Estate": "XLRE",
    "Software & Svcs": "IGV",
    "Tech Hardware & Equip": "IYW",
    "Semis & Equip": "SOXX",
    "Telecom Svcs": "IYZ",
    "Media & Entertainment": "PBS",
    "Utilities": "XLU",
}
start = "2023-05-20"  # 1-yr look-back
end = "2025-05-20"
# ---------------------------------------------------------------------------

# download Adj Close prices
assets = yf.download(list(available_tickers.values()), start=start, end=end)

assets_cols = set(assets.columns.get_level_values(0))
print(f"Available columns in yf assets: {assets_cols}")

asset_close = assets["Close"]
asset_close.columns = available_tickers.keys()
# -

pct_returns = assets["Close"].pct_change().apply(lambda x: np.log1p(x)).dropna()
