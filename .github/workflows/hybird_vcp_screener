# -*- coding: utf-8 -*-
"""
Hybrid VCP Screener for S&P 500 — Enhanced Edition
- Strict & Practical VCP with improved RS calculation
- Market filter (SPY above 200‑day MA)
- Volume liquidity threshold
- Telegram message auto-splitting (≤4000 chars)
- Centralised configuration
- Fallback to CSV output if openpyxl is missing
"""

import io
import os
import html
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import certifi
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- Configuration -----------------------------
@dataclass
class Config:
    """All tunable parameters in one place."""
    # Data range
    history_days: int = 420                 # lookback for RS calculation
    rs_ma_days: int = 200                   # for market filter
    min_volume: int = 100_000               # minimal 50‑day avg volume
    rs_percentile: float = 70.0             # top RS percentile to keep

    # VCP detection
    lookback_contractions: int = 160        # max bars for contraction search
    order_strict: int = 5
    order_practical: int = 4
    min_pct: float = 0.03
    max_pct_strict: float = 0.30
    max_pct_practical: float = 0.35
    dryup_ratio_strict: float = 0.70        # last pullback vol / avg50 < this
    dryup_ratio_practical: float = 0.80
    near_pivot_pct: float = 5.0             # distance to pivot to be "near"
    breakout_vol_mult: float = 1.4

    # Output
    output_dir: str = 'output'
    output_excel: str = 'Hybrid_VCP_ScreenOutput.xlsx'

    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))

config = Config()

# ----------------------------- Helpers ------------------------------------
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def output_dir():
    out = os.path.join(script_dir(), config.output_dir)
    os.makedirs(out, exist_ok=True)
    return out

INDEX_SYMBOL = '^GSPC'
WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
CSV_FALLBACK_URL = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_sp500_tickers():
    errors = []
    try:
        resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30, verify=certifi.where())
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        tickers = table['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
        if tickers:
            return tickers
    except Exception as e:
        errors.append(f'Wikipedia failed: {e}')

    try:
        resp = requests.get(CSV_FALLBACK_URL, headers=HEADERS, timeout=30, verify=certifi.where())
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        tickers = df['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
        if tickers:
            return tickers
    except Exception as e:
        errors.append(f'GitHub fallback failed: {e}')

    raise RuntimeError('Unable to retrieve S&P 500 tickers. ' + ' | '.join(errors))


def get_price_df(symbol, start=None, end=None):
    """Download price data and return a clean DataFrame with OHLCV columns."""
    if start is None:
        start = (datetime.now() - timedelta(days=config.history_days)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return None

    # Handle MultiIndex columns (newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract the ticker-specific data
        if symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, axis=1, level=1)
        else:
            # fallback: use first available security
            df.columns = df.columns.get_level_values(0)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            return None

    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    return df.dropna().copy()


def market_ok(df_spy):
    """Return True if SPY is above its 200-day moving average."""
    if df_spy is None or len(df_spy) < config.rs_ma_days:
        return False
    spy_close = df_spy['Adj Close']
    ma200 = spy_close.rolling(window=config.rs_ma_days).mean().iloc[-1]
    return pd.notna(ma200) and spy_close.iloc[-1] > ma200


# ----------------------------- RS & Trend --------------------------------
def compute_rs(symbol, index_df):
    """Compute RS multiple using common trading days."""
    df = get_price_df(symbol)
    if df is None or len(df) < 50:
        return None, None

    # Align dates
    common_dates = df.index.intersection(index_df.index)
    if len(common_dates) < 50:
        return None, None

    stock_close = df.loc[common_dates, 'Adj Close']
    spx_close = index_df.loc[common_dates, 'Adj Close']

    stock_return = stock_close.iloc[-1] / stock_close.iloc[0]
    spx_return = spx_close.iloc[-1] / spx_close.iloc[0]
    rs_multiple = stock_return / spx_return
    return rs_multiple, df


def local_extrema(series, order=4):
    highs, lows = [], []
    vals = series.values
    idx = series.index
    for i in range(order, len(series) - order):
        window = vals[i - order:i + order + 1]
        center = vals[i]
        if np.isfinite(center) and center == np.max(window):
            highs.append((idx[i], float(center), i))
        if np.isfinite(center) and center == np.min(window):
            lows.append((idx[i], float(center), i))
    return highs, lows


def extract_contractions(df, order, max_lookback, min_pct, max_pct):
    recent = df.tail(max_lookback).copy()
    highs, lows = local_extrema(recent['Adj Close'], order=order)
    contractions = []

    for h_date, h_price, h_pos in highs:
        next_lows = [(d, p, pos) for d, p, pos in lows if pos > h_pos]
        if not next_lows:
            continue
        l_date, l_price, l_pos = next_lows[0]
        days = l_pos - h_pos
        if days < 3:
            continue
        pct = (h_price - l_price) / h_price
        if min_pct <= pct <= max_pct:
            avg_vol = recent['Volume'].iloc[h_pos:l_pos + 1].mean()
            contractions.append({
                'high_date': h_date,
                'low_date': l_date,
                'high_pos': h_pos,
                'low_pos': l_pos,
                'high_price': h_price,
                'low_price': l_price,
                'contraction_pct': pct,
                'avg_pullback_volume': avg_vol,
            })

    return contractions[-4:], recent


def trend_template(df, strict=False):
    for x in [20, 50, 150, 200]:
        df[f'SMA_{x}'] = df['Adj Close'].rolling(window=x).mean()

    close = float(df['Adj Close'].iloc[-1])
    ma20 = float(df['SMA_20'].iloc[-1])
    ma50 = float(df['SMA_50'].iloc[-1])
    ma150 = float(df['SMA_150'].iloc[-1])
    ma200 = float(df['SMA_200'].iloc[-1])
    ma200_20 = float(df['SMA_200'].iloc[-20]) if len(df) >= 220 and pd.notna(df['SMA_200'].iloc[-20]) else np.nan
    high_52w = float(df['High'].tail(250).max())
    low_52w = float(df['Low'].tail(250).min())

    if strict:
        ok = all([
            pd.notna(ma20), pd.notna(ma50), pd.notna(ma150), pd.notna(ma200),
            close > ma50 > ma150 > ma200,
            ma150 > ma200,
            pd.notna(ma200_20) and ma200 > ma200_20,
            close >= 0.80 * high_52w,
            close >= 1.30 * low_52w,
            close > 10,
        ])
    else:
        ok = all([
            pd.notna(ma50), pd.notna(ma150), pd.notna(ma200),
            close > ma50 > ma150 > ma200,
            ma150 > ma200,
            pd.notna(ma200_20) and ma200 >= ma200_20,
            close >= 0.75 * high_52w,
            close >= 1.25 * low_52w,
            close > 10,
        ])

    return {
        'ok': ok,
        'close': close,
        'ma20': ma20,
        'ma50': ma50,
        'ma150': ma150,
        'ma200': ma200,
        'high_52w': high_52w,
        'low_52w': low_52w,
    }


# ----------------------------- VCP Evaluators (enhanced) -----------------
def evaluate_strict_vcp(df):
    if df is None or len(df) < 250:
        return None

    t = trend_template(df, strict=True)
    if not t['ok']:
        return None

    contractions, recent = extract_contractions(
        df, order=config.order_strict,
        max_lookback=config.lookback_contractions,
        min_pct=config.min_pct,
        max_pct=config.max_pct_strict
    )

    if len(contractions) < 2:
        return None

    contraction_pcts = [c['contraction_pct'] for c in contractions]
    # Contraction size must shrink (each subsequent smaller)
    if not all(contraction_pcts[i] > contraction_pcts[i + 1] for i in range(len(contraction_pcts) - 1)):
        return None

    volume_seq = [c['avg_pullback_volume'] for c in contractions]
    avg50_volume = float(recent['Volume'].tail(50).mean())
    if avg50_volume < config.min_volume:
        return None

    last = contractions[-1]
    # Enhanced: last pullback volume must be significantly below average
    last_vol_ratio = volume_seq[-1] / avg50_volume
    if last_vol_ratio > config.dryup_ratio_strict:
        return None

    # Also ensure volume trend is generally decreasing (>= half of consecutive pairs)
    decreasing_pairs = sum(1 for i in range(len(volume_seq)-1) if volume_seq[i] >= volume_seq[i+1])
    if decreasing_pairs < max(1, len(volume_seq) - 2):
        return None

    pivot = float(recent['High'].iloc[last['low_pos']:].max())
    latest_close = float(recent['Adj Close'].iloc[-1])
    latest_volume = float(recent['Volume'].iloc[-1])
    distance_to_pivot_pct = ((pivot - latest_close) / pivot) * 100 if pivot > 0 else np.nan
    near_pivot = latest_close >= pivot * 0.97
    breakout_now = (latest_close > pivot) and (latest_volume >= config.breakout_vol_mult * avg50_volume)

    setup_type = 'Strict VCP'
    if breakout_now:
        setup_type = 'Strict Breakout'
    elif near_pivot:
        setup_type = 'Strict Near Pivot'

    return {
        'Mode': 'Strict',
        'Setup Type': setup_type,
        'Close': round(t['close'], 2),
        'Pivot': round(pivot, 2),
        'Distance to Pivot %': round(distance_to_pivot_pct, 2) if pd.notna(distance_to_pivot_pct) else np.nan,
        'Near Pivot': bool(near_pivot),
        'Breakout Now': bool(breakout_now),
        'Contractions': ' | '.join(f'{x * 100:.1f}%' for x in contraction_pcts),
        'Volume Dry-Up Ratio': round(last_vol_ratio, 2),
        'Avg Pullback Volumes': ' | '.join(f'{int(v):,}' for v in volume_seq),
        '52W High': round(t['high_52w'], 2),
        '52W Low': round(t['low_52w'], 2),
        '50MA': round(t['ma50'], 2),
        '150MA': round(t['ma150'], 2),
        '200MA': round(t['ma200'], 2),
    }


def evaluate_practical_vcp(df):
    if df is None or len(df) < 250:
        return None

    t = trend_template(df, strict=False)
    if not t['ok']:
        return None

    contractions, recent = extract_contractions(
        df, order=config.order_practical,
        max_lookback=config.lookback_contractions,
        min_pct=config.min_pct,
        max_pct=config.max_pct_practical
    )

    if len(contractions) < 2:
        return None

    contraction_pcts = [c['contraction_pct'] for c in contractions]
    first_last_tighter = contraction_pcts[-1] < contraction_pcts[0]
    non_expanding_count = sum(contraction_pcts[i + 1] <= contraction_pcts[i] for i in range(len(contraction_pcts) - 1))
    if not (first_last_tighter and non_expanding_count >= max(1, len(contraction_pcts) - 2)):
        return None

    volume_seq = [c['avg_pullback_volume'] for c in contractions]
    avg50_volume = float(recent['Volume'].tail(50).mean())
    if avg50_volume < config.min_volume:
        return None

    last_vol = volume_seq[-1]
    dryup_ratio = last_vol / avg50_volume
    if not (pd.notna(dryup_ratio) and dryup_ratio <= config.dryup_ratio_practical):
        return None

    last = contractions[-1]
    pivot = float(recent['High'].iloc[last['low_pos']:].max())
    latest_close = float(recent['Adj Close'].iloc[-1])
    latest_volume = float(recent['Volume'].iloc[-1])
    distance_to_pivot_pct = ((pivot - latest_close) / pivot) * 100 if pivot > 0 else np.nan
    near_pivot = pd.notna(distance_to_pivot_pct) and 0 <= distance_to_pivot_pct <= config.near_pivot_pct
    breakout_now = latest_close > pivot and latest_volume >= config.breakout_vol_mult * avg50_volume

    setup_type = 'Watchlist'
    if breakout_now:
        setup_type = 'Breakout Today'
    elif near_pivot:
        setup_type = 'Near Pivot'

    return {
        'Mode': 'Practical',
        'Setup Type': setup_type,
        'Close': round(t['close'], 2),
        'Pivot': round(pivot, 2),
        'Distance to Pivot %': round(distance_to_pivot_pct, 2) if pd.notna(distance_to_pivot_pct) else np.nan,
        'Near Pivot': bool(near_pivot),
        'Breakout Now': bool(breakout_now),
        'Contractions': ' | '.join(f'{x * 100:.1f}%' for x in contraction_pcts),
        'Volume Dry-Up Ratio': round(dryup_ratio, 2),
        'Avg Pullback Volumes': ' | '.join(f'{int(v):,}' for v in volume_seq),
        '52W High': round(t['high_52w'], 2),
        '52W Low': round(t['low_52w'], 2),
        '50MA': round(t['ma50'], 2),
        '150MA': round(t['ma150'], 2),
        '200MA': round(t['ma200'], 2),
    }


# ----------------------------- Telegram helpers -------------------------
def split_long_message(text: str, max_len: int = 4000) -> List[str]:
    """Split message into chunks that respect HTML tag boundaries if possible."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    lines = text.split('\n')
    current = ''
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current += '\n' + line if current else line
    if current:
        chunks.append(current)
    return chunks


def format_telegram_message(strict_df, practical_df, combined_df):
    if combined_df.empty:
        return '<b>Hybrid VCP Screen</b>\nNo setups found today.'

    lines = ['<b>Hybrid VCP Screen</b>']
    lines.append(f"<b>Total:</b> {len(combined_df)} | <b>Strict:</b> {len(strict_df)} | <b>Practical:</b> {len(practical_df)}")

    sections = [
        ('🔥 Strict Breakout', strict_df[strict_df['Setup Type'] == 'Strict Breakout'].head(5)),
        ('🎯 Strict Near Pivot', strict_df[strict_df['Setup Type'].isin(['Strict Near Pivot', 'Strict VCP'])].head(5)),
        ('🚀 Practical Breakout', practical_df[practical_df['Setup Type'] == 'Breakout Today'].head(5)),
        ('👀 Practical Near Pivot', practical_df[practical_df['Setup Type'] == 'Near Pivot'].head(5)),
        ('📌 Practical Watchlist', practical_df[practical_df['Setup Type'] == 'Watchlist'].head(5)),
    ]

    for title, subset in sections:
        if subset.empty:
            continue
        lines.append(f"\n<b>{html.escape(title)}</b>")
        for _, row in subset.iterrows():
            stock = html.escape(str(row['Stock']))
            contractions = html.escape(str(row['Contractions']))
            dist = row['Distance to Pivot %']
            dry = row['Volume Dry-Up Ratio']
            lines.append(
                f"• <b>{stock}</b> | RS {row['RS_Rating']:.0f} | Pivot {row['Pivot']:.2f} | "
                f"Dist {dist:.2f}% | DryUp {dry:.2f} | {contractions}"
            )

    return '\n'.join(lines)


def send_telegram_messages(messages: List[str]):
    """Send each chunk as a separate message."""
    for msg in messages:
        if not config.bot_token or not config.chat_id:
            print('Telegram credentials not set, skipping.')
            return
        url = f'https://api.telegram.org/bot{config.bot_token}/sendMessage'
        payload = {
            'chat_id': config.chat_id,
            'text': msg,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        print(f'Telegram chunk sent (length {len(msg)})')


# ----------------------------- Main workflow -----------------------------
def main():
    os.makedirs(output_dir(), exist_ok=True)

    # 1. Get S&P 500 tickers and SPY data
    tickers = get_sp500_tickers()
    index_df = get_price_df(INDEX_SYMBOL)
    if index_df is None or index_df.empty:
        raise ValueError('Failed to download S&P 500 index data.')

    # 2. Market filter (SPY above 200‑MA)
    if not market_ok(index_df):
        msg = '<b>Hybrid VCP Screen</b>\n❌ Market filter: SPY below 200-day MA. No signals today.'
        send_telegram_messages([msg])
        print('Market filter blocked. SPY below 200MA.')
        return

    # 3. Compute RS for all stocks
    rs_rows = []
    cache = {}
    for ticker in tickers:
        try:
            rs_multiple, df = compute_rs(ticker, index_df)
            if rs_multiple is None or df is None:
                continue

            # Volume filter
            avg_vol = df['Volume'].rolling(50).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol < config.min_volume:
                continue

            rs_rows.append({'Ticker': ticker, 'RS_Multiple': rs_multiple})
            cache[ticker] = df
            print(f'{ticker}: RS={rs_multiple:.2f}')
        except Exception as e:
            print(f'Error {ticker}: {e}')

    if not rs_rows:
        print('No stocks passed RS and volume filters.')
        return

    rs_df = pd.DataFrame(rs_rows)
    rs_df['RS_Rating'] = rs_df['RS_Multiple'].rank(pct=True) * 100
    rs_df = rs_df[rs_df['RS_Rating'] >= config.rs_percentile].copy()
    rs_df = rs_df.sort_values('RS_Rating', ascending=False)

    # 4. Evaluate VCP patterns
    strict_rows, practical_rows = [], []
    for stock in rs_df['Ticker']:
        df = cache.get(stock)
        rs_val = rs_df.loc[rs_df['Ticker'] == stock, 'RS_Rating'].iloc[0]

        strict_res = evaluate_strict_vcp(df)
        if strict_res is not None:
            strict_rows.append({'Stock': stock, 'RS_Rating': round(rs_val, 2), **strict_res})

        practical_res = evaluate_practical_vcp(df)
        if practical_res is not None:
            practical_rows.append({'Stock': stock, 'RS_Rating': round(rs_val, 2), **practical_res})

    # 5. Build DataFrames
    all_cols = ['Stock', 'RS_Rating', 'Mode', 'Setup Type', 'Close', 'Pivot',
                'Distance to Pivot %', 'Near Pivot', 'Breakout Now', 'Contractions',
                'Volume Dry-Up Ratio', 'Avg Pullback Volumes', '52W High', '52W Low',
                '50MA', '150MA', '200MA']

    strict_df = pd.DataFrame(strict_rows, columns=all_cols) if strict_rows else pd.DataFrame(columns=all_cols)
    practical_df = pd.DataFrame(practical_rows, columns=all_cols) if practical_rows else pd.DataFrame(columns=all_cols)

    combined_df = pd.concat([strict_df, practical_df], ignore_index=True) if not (strict_df.empty or practical_df.empty) else \
                   strict_df if not strict_df.empty else practical_df

    if not combined_df.empty:
        priority = {
            'Strict Breakout': 0,
            'Strict Near Pivot': 1,
            'Strict VCP': 2,
            'Breakout Today': 3,
            'Near Pivot': 4,
            'Watchlist': 5,
        }
        combined_df['SetupRank'] = combined_df['Setup Type'].map(priority).fillna(9)
        combined_df = combined_df.sort_values(['SetupRank', 'RS_Rating', 'Distance to Pivot %'],
                                              ascending=[True, False, True]).drop(columns=['SetupRank'])
        strict_df = strict_df.sort_values(['RS_Rating', 'Distance to Pivot %'], ascending=[False, True])
        practical_df = practical_df.sort_values(['RS_Rating', 'Distance to Pivot %'], ascending=[False, True])

    # 6. Write to Excel (or CSV if openpyxl not available)
    try:
        import openpyxl
        excel_path = os.path.join(output_dir(), config.output_excel)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, index=False, sheet_name='Combined')
            strict_df.to_excel(writer, index=False, sheet_name='Strict VCP')
            practical_df.to_excel(writer, index=False, sheet_name='Practical VCP')
            if not practical_df.empty:
                practical_df[practical_df['Setup Type'] == 'Watchlist'].to_excel(writer, index=False, sheet_name='Watchlist')
                practical_df[practical_df['Setup Type'] == 'Near Pivot'].to_excel(writer, index=False, sheet_name='Near Pivot')
                practical_df[practical_df['Setup Type'] == 'Breakout Today'].to_excel(writer, index=False, sheet_name='Breakout Today')
        print(f'Excel saved: {excel_path}')
    except ImportError:
        print('openpyxl not installed – writing CSV files instead.')
        combined_df.to_csv(os.path.join(output_dir(), 'Combined.csv'), index=False)
        strict_df.to_csv(os.path.join(output_dir(), 'Strict_VCP.csv'), index=False)
        practical_df.to_csv(os.path.join(output_dir(), 'Practical_VCP.csv'), index=False)

    # 7. Send Telegram (chunked if necessary)
    msg = format_telegram_message(strict_df, practical_df, combined_df)
    chunks = split_long_message(msg, max_len=4000)
    try:
        send_telegram_messages(chunks)
    except Exception as e:
        print(f'Telegram send failed: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
