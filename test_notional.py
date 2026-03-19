import yfinance as yf
import pandas as pd

def get_notional_price_roe_judgement(ticker_symbol, required_return=0.10, max_growth=0.20, max_roe=0.30):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        eps = info.get("trailingEps")
        roe = info.get("returnOnEquity")
        payout = info.get("payoutRatio")
        current_price = info.get("currentPrice")
        
        if eps is None or eps <= 0:
            return f"{ticker_symbol}: No usable EPS data"
        
        if roe is None or roe <= 0:
            return f"{ticker_symbol}: No usable ROE data"
        
        # Clean payout ratio
        if payout is None or payout < 0:
            payout = 0
        payout = min(max(payout, 0), 0.9)
        
        # Cap ROE for justified P/E purposes
        adjusted_roe = min(roe, max_roe)
        
        # Sustainable growth
        g = adjusted_roe * (1 - payout)
        g = min(g, max_growth)
        
        # Adjust denominator to avoid tiny values
        denominator = max(required_return - g, 0.01)
        
        justified_pe = (adjusted_roe * (1 - payout)) / denominator
        fair_value = eps * justified_pe
        entry_price = fair_value * 0.8  # 20% margin of safety
        
        # Valuation ratio and judgement
        valuation_ratio = current_price / fair_value
        if valuation_ratio < 0.9:
            judgement = "Undervalued"
        elif valuation_ratio <= 1.1:
            judgement = "Fairly valued"
        else:
            judgement = "Overvalued"
        
        return {
            "Ticker": ticker_symbol,
            "Current Price": current_price,
            "EPS": round(eps, 2),
            "ROE": round(roe * 100, 2),
            "Payout Ratio": round(payout * 100, 2),
            "Adjusted ROE (%)": round(adjusted_roe * 100, 2),
            "Growth Used (%)": round(g * 100, 2),
            "Justified P/E": round(justified_pe, 2),
            "Fair Value": round(fair_value, 2),
            "Notional Entry Price": round(entry_price, 2),
            "Valuation": round(valuation_ratio, 2),
            "Judgement": judgement
        }
        
    except Exception as e:
        return f"Error processing {ticker_symbol}: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute notional share price (ROE-based) for given tickers.")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols (e.g. AAPL MSFT NVDA)")
    parser.add_argument("--required-return", "-r", type=float, default=0.10, help="Required return (default 0.10)")
    parser.add_argument("--max-growth", "-g", type=float, default=0.20, help="Max growth cap (default 0.20)")
    parser.add_argument("--max-roe", type=float, default=0.30, help="Max ROE cap (default 0.30)")
    args = parser.parse_args()

    results = []
    for t in args.tickers:
        out = get_notional_price_roe_judgement(
            t.strip().upper(),
            required_return=args.required_return,
            max_growth=args.max_growth,
            max_roe=args.max_roe,
        )
        results.append(out if isinstance(out, dict) else {"Ticker": t, "Error": out})

    df = pd.DataFrame(results)
    print(df)
