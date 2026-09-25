"""
indicator_data_pull_options.py -- pulls NIFTY INDEX OPTIONS daily bhavcopy
data from NSE for the OOS window (2017-01-01 onward), needed to properly
test put/call option hedges (as opposed to the linear futures-style
approximations already tested in hedge_fraction_sweep_v2.py, which is all
that could be built without this data).

RUN THIS ON YOUR OWN MACHINE, NOT IN THIS SESSION: the sandbox this
conversation runs in has archives.nseindia.com AND nsearchives.nseindia.com
BLOCKED at the network policy level (confirmed directly -- not a transient
failure, the proxy reports a policy denial, and policy denials are not
something to route around). Same reason indicator_data_pull_v2.py/_v3.py
were built for you to run locally in earlier rounds of this project.

TWO NSE FORMATS, NOT ONE -- NSE changed the bhavcopy format on 2024-07-08.
This script handles both, sourced (not guessed) via a live web search of
NSE's own published format spec and corroborating public parsers:
  LEGACY (dates <= 2024-07-05):
    https://archives.nseindia.com/content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
    columns used: SYMBOL, INSTRUMENT, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
    CLOSE, SETTLE_PR, OPEN_INT, CONTRACTS, TIMESTAMP
    index options rows: INSTRUMENT == 'OPTIDX'
  UDiFF (dates >= 2024-07-08):
    https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
    columns used: TckrSymb, FinInstrmTp, XpryDt, StrkPric, OptnTp, ClsPric,
    SttlmPric, OpnIntrst, TtlTradgVol, TradDt
    index options rows: FinInstrmTp is EXPECTED to be 'IDO' (SEBI's standard
    code for Index Options) based on the general SEBI/NSE derivatives
    segment coding convention (IDF/IDO/STF/STO) -- NOT verified against a
    real downloaded file (blocked from doing so in this session). THE
    SCRIPT PRINTS every unique FinInstrmTp value it actually finds on the
    first UDiFF-era file it downloads, specifically so this assumption gets
    checked against real data on your very first run rather than silently
    trusted -- if 'IDO' isn't among the printed values, tell me what IS and
    I'll fix the filter.

OUTPUT: one row per (date, expiry, strike, option_type) for NIFTY index
options only (filtered down from the full bhavcopy, which includes every
stock's and every index's derivatives, before saving -- keeps output size
manageable). Appended to data/pit/nifty_options_daily.csv, checkpointed by
date so a re-run picks up where it left off rather than re-downloading
everything.

Uses the same checkpointing / retry-with-backoff SHAPE as
indicator_data_pull_v2.py, but NOT its Chrome-fallback path -- that path
fetches small JSON/CSV text endpoints through a browser's page source,
which doesn't translate to downloading a binary .zip file, so it isn't
reused here (no dead import pretending otherwise). NSE's archive downloads
still need a real browser-like session (cookies set by visiting the site
first), handled via a requests.Session() primed with a homepage visit
before hitting the archive URLs. If that alone isn't enough to get past
NSE's bot checks, tell me and a real Chrome-download fallback (Selenium,
polling a download directory for the .zip -- a different mechanism than
v2's text-endpoint trick) can be built, but it isn't in this version since
it's real added complexity I haven't confirmed you need yet.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import warnings
import zipfile

warnings.filterwarnings("ignore")

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PIT = os.path.join(DATA, "pit")
os.makedirs(PIT, exist_ok=True)

OUT_FILE = os.path.join(PIT, "nifty_options_daily.csv")
CHECKPOINT_FILE = os.path.join(PIT, "nifty_options_daily.checkpoint.txt")
UDIFF_CUTOVER = pd.Timestamp("2024-07-08")
START_DATE = pd.Timestamp("2017-01-01")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

_printed_udiff_instrument_types = False


def _nse_session() -> requests.Session:
    """NSE's archive endpoints reject bare requests -- a session primed by
    visiting the homepage first (to pick up cookies) is the standard,
    documented workaround every public NSE-data tool uses."""
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except Exception:
        pass
    return s


def legacy_url(date: pd.Timestamp) -> str:
    mon = date.strftime("%b").upper()
    return (f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
           f"{date.year}/{mon}/fo{date.strftime('%d')}{mon}{date.year}bhav.csv.zip")


def udiff_url(date: pd.Timestamp) -> str:
    return (f"https://archives.nseindia.com/content/fo/"
           f"BhavCopy_NSE_FO_0_0_0_{date.strftime('%Y%m%d')}_F_0000.csv.zip")


def _download_zip_csv(session: requests.Session, url: str) -> pd.DataFrame | None:
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = z.namelist()
            if not names:
                return None
            with z.open(names[0]) as f:
                return pd.read_csv(f)
    except Exception as e:
        print(f"    fetch failed: {str(e)[:80]}")
        return None


def parse_legacy(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    # NSE's legacy bhavcopy CSVs are known to pad string values with trailing
    # spaces (e.g. "NIFTY   ") -- a plain == match against them silently
    # returns zero rows even though the data is there. Strip before matching.
    symbol = df["SYMBOL"].astype(str).str.strip()
    instrument = df["INSTRUMENT"].astype(str).str.strip()
    sub = df[(symbol == "NIFTY") & (instrument == "OPTIDX")].copy()
    if sub.empty:
        return sub
    out = pd.DataFrame({
        "trade_date": date, "expiry_date": pd.to_datetime(sub["EXPIRY_DT"], errors="coerce"),
        "strike": sub["STRIKE_PR"].astype(float),
        "option_type": sub["OPTION_TYP"].astype(str).str.strip(),
        "close": sub["CLOSE"].astype(float), "settle_price": sub["SETTLE_PR"].astype(float),
        "open_interest": sub["OPEN_INT"].astype(float), "volume": sub["CONTRACTS"].astype(float),
        "format": "legacy",
    })
    return out


def parse_udiff(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    global _printed_udiff_instrument_types
    df.columns = [c.strip() for c in df.columns]
    if not _printed_udiff_instrument_types:
        print(f"    [CHECK] unique FinInstrmTp values in this UDiFF file: "
              f"{sorted(df['FinInstrmTp'].astype(str).str.strip().dropna().unique().tolist())}")
        print(f"    [CHECK] confirm 'IDO' (or whichever code NIFTY index options actually use) "
              f"is among these -- if not, tell me the real one and this filter gets fixed.")
        _printed_udiff_instrument_types = True

    idx_option_codes = {"IDO"}  # see module docstring -- verify against the printed [CHECK] line above
    ticker = df["TckrSymb"].astype(str).str.strip()
    instrument_type = df["FinInstrmTp"].astype(str).str.strip()
    sub = df[(ticker == "NIFTY") & (instrument_type.isin(idx_option_codes))].copy()
    if sub.empty:
        return sub
    out = pd.DataFrame({
        "trade_date": date, "expiry_date": pd.to_datetime(sub["XpryDt"], errors="coerce"),
        "strike": sub["StrkPric"].astype(float),
        "option_type": sub["OptnTp"].astype(str).str.strip(),
        "close": sub["ClsPric"].astype(float), "settle_price": sub["SttlmPric"].astype(float),
        "open_interest": sub["OpnIntrst"].astype(float), "volume": sub["TtlTradgVol"].astype(float),
        "format": "udiff",
    })
    return out


def already_have(date_str: str) -> bool:
    if not os.path.exists(CHECKPOINT_FILE):
        return False
    with open(CHECKPOINT_FILE) as f:
        done = set(line.strip() for line in f)
    return date_str in done


def mark_done(date_str: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(date_str + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_DATE.strftime("%Y-%m-%d"))
    ap.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()

    trading_days = pd.bdate_range(args.start, args.end)  # weekday approx; NSE holidays just get empty/failed responses, skipped
    print(f"NIFTY index options pull: {trading_days[0].date()} -> {trading_days[-1].date()} "
         f"({len(trading_days)} candidate weekdays)")
    print(f"Legacy format through 2024-07-05, UDiFF format from 2024-07-08 onward")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")

    session = _nse_session()
    n_saved, n_skipped, n_failed, n_empty = 0, 0, 0, 0

    for date in trading_days:
        date_str = date.strftime("%Y-%m-%d")
        if already_have(date_str):
            n_skipped += 1
            continue

        url = legacy_url(date) if date < UDIFF_CUTOVER else udiff_url(date)
        parser = parse_legacy if date < UDIFF_CUTOVER else parse_udiff

        df = None
        for attempt in range(4):
            raw = _download_zip_csv(session, url)
            if raw is not None:
                df = parser(raw, date)
                break
            print(f"  {date_str}: attempt {attempt + 1}/4 failed, retrying in 15s...")
            time.sleep(15)
            session = _nse_session()  # re-prime cookies

        if df is None:
            print(f"  {date_str}: FAILED after 4 attempts (likely a holiday, or genuinely blocked -- "
                 f"checkpointed as done either way so a re-run doesn't loop on it forever; "
                 f"re-run with --start {date_str} if you want to retry it specifically)")
            n_failed += 1
        elif df.empty:
            print(f"  {date_str}: no NIFTY index option rows found (probably a market holiday)")
            n_empty += 1
        else:
            df.to_csv(OUT_FILE, mode="a", header=not os.path.exists(OUT_FILE), index=False)
            n_saved += 1
            if n_saved % 50 == 0:
                print(f"  ... {n_saved} days saved so far (latest: {date_str}, {len(df)} rows)")

        mark_done(date_str)
        time.sleep(0.5)  # be a bit gentle on NSE's archive server

    print(f"\nDone. saved={n_saved} skipped(already had)={n_skipped} "
         f"failed/holiday={n_failed} empty={n_empty}")
    print(f"Output: {OUT_FILE}")
    print(f"Send that file back, plus this console output (especially any [CHECK] lines) so I can "
         f"confirm the UDiFF instrument-type filter matched real data correctly.")


if __name__ == "__main__":
    main()
