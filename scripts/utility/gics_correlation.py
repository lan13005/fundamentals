import yfinance as yf
import pandas as pd
import numpy as np

def main():
    # ---- user parameters -------------------------------------------------------
    tickers = {
        'Energy':'XLE','Materials':'XLB','Capital Goods':'IYJ',
        'Comm & Prof Svcs':'XLI','Transportation':'IYT',
        'Autos & Components':'CARZ','Cons Durables & Apparel':'IBUY',
        'Consumer Services':'PEJ','Retailing':'XRT',
        'Food & Staples Retail':'XLP','Food/Beverage/Tobacco':'PBJ',
        'Household & Pers Prod':'IYK','HC Equip & Svcs':'IHI',
        'Pharma / Biotech':'XPH','Banks':'KBE',
        'Diversified Financials':'IYF','Insurance':'KIE',
        'Real Estate':'XLRE','Software & Svcs':'IGV',
        'Tech Hardware & Equip':'IYW','Semis & Equip':'SOXX',
        'Telecom Svcs':'IYZ','Media & Entertainment':'PBS',
        'Utilities':'XLU'
    }
    start  = "2024-05-20"   # 1-yr look-back
    end    = "2025-05-20"
    # ---------------------------------------------------------------------------

    # download Adj Close prices
    prices = yf.download(list(tickers.values()), start=start, end=end)['Adj Close']
    prices.columns = tickers.keys()

    # daily log-returns
    rets = prices.pct_change().apply(lambda x: np.log1p(x))

    # correlation matrix
    corr = rets.corr().round(2)

    # pretty print
    pd.set_option('display.max_columns', None)
    print(corr)

    # optional: save
    corr.to_csv(f"gics_lvl2_corr_{start}to{end}.csv")
