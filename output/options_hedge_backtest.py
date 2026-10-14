"""
options_hedge_backtest.py -- tests REAL put-option and covered-call hedge
structures against actual NIFTY options premiums (data/pit/nifty_options_daily.csv),
replacing the linear-return proxies used for hedge_nifty_short/hedge_vixy in
hedge_fraction_sweep_v2.py. Built directly from the question "i think you
dont have the call and put option data you need to verify that right?" --
correct, and now that data exists, this uses it for real.

Reuses the TESTED equity-leg mechanics (core-slice sale, blended LTCG/STCG
tax, cost-basis tracking, block-finding) from hedge_fraction_sweep_v2.py via
import, unchanged. Only the OPTIONS mechanics below are new.

TWO STRUCTURALLY DIFFERENT VARIANTS, both explicitly asked about in chat
("what about nifty shorts, or put option even write calls option?"):

  1. LONG PUT ("hedge_put_atm", "hedge_put_otm5") -- the carved-out slice
     buys NIFTY PUTS (ATM, or 5% out-of-the-money) at the nearest listed
     expiry, rolled forward at each expiry until the Rule-B hold window
     ends, then closed. Real premium paid, real time decay, real payoff --
     this is the genuine version of what hedge_nifty_short was a linear
     stand-in for.

  2. COVERED CALL ("hedge_covered_call") -- structurally DIFFERENT, tested
     because it was explicitly asked about, with the expected result
     stated up front rather than hidden: standard options theory says this
     does NOT hedge a crash. The underlying is never sold (no equity sale,
     no equity tax event) -- an ATM call is written against the full
     position, collecting premium as a small cushion, capping upside, but
     leaving nearly all downside exposure intact beyond that premium. This
     script measures that outcome rather than assumes it.

OPTIONS-SPECIFIC TRANSACTION COSTS, sourced via WebSearch (2026-current
rates, NOT the futures-calibrated fo_buy_cost/fo_sell_cost already in v2 --
options STT/stamp rates are genuinely different from futures'):
  - STT: 0% on buy-to-open (options are STT-free on the buy side, unlike
    equity). 0.15% of PREMIUM on any sell-to-close (closing a long put
    early, or writing/selling a call) -- current rate after the April 2026
    STT unification (was 0.10% Oct 2024-Mar 2026, 0.0625% before that;
    this project uses the current rate uniformly across the whole
    2017-2026 backtest, same simplification already used for equity costs
    elsewhere here, not attempting a year-by-year historical rate
    reconstruction). 0.15% of INTRINSIC VALUE on cash settlement of an ITM
    long option at expiry -- a different base than a sell, correctly kept
    separate (these two rates happened to converge to the same number in
    the April 2026 change, but the mechanism is genuinely distinct and is
    modeled as such rather than collapsed into one constant by coincidence).
  - Exchange transaction charge: ~0.035% of premium value (NSE index
    options rate, both legs).
  - Stamp duty: 0.003% of premium, buy side only (SEBI's standardized 2020
    schedule -- the OPTIONS rate, distinct from the 0.002% FUTURES rate
    already used for hedge_vixy/hedge_nifty_short in v2).
  - Brokerage: flat Rs 20/executed order (Groww, confirmed unchanged for
    options vs. futures).
  - GST: 18% on (brokerage + exchange charge); stamp duty is a state tax,
    not GST'd -- same convention brokers actually use.
  - Disclosed simplification: no separate brokerage/exchange charge is
    modeled on final ITM expiry cash settlement itself (most discount
    brokers, Groww included, don't charge brokerage on auto square-off of
    a cash-settled index option at expiry) -- only the exercise STT is
    charged there. A small, disclosed approximation, not a hidden one.
  Sources: STT history/current level -- Zerodha Z-Connect
  (https://zerodha.com/z-connect/business-updates/revision-in-exchange-transaction-charges-and-securities-transaction-tax-from-october-1-2024),
  Finnovate's Budget-2026 explainer
  (https://www.finnovate.in/learn/blog/budget-2026-stt-hike-fno-trades-explained).
  Stamp duty / exchange charge schedule -- Zerodha charges page
  (https://zerodha.com/charges/), NSE stamp duty page
  (https://www.nseindia.com/static/invest/first-time-investor-stamp-duty-charges-taxes).

A REAL DATA QUALITY ISSUE FOUND AND HANDLED, not silently worked around:
  319,005 of 5,115,945 rows (6.2%) in the pulled NSE archive have a BLANK
  expiry_date in NSE's own published legacy-era bhavcopy file, concentrated
  on 164 specific historical trading days (2017-2024) where most of that
  day's contracts are affected -- confirmed this is present in the raw
  archive itself (not introduced by the CSV split/reassembly, which was
  separately verified byte-exact). These rows are EXCLUDED from the option
  chain used here (a contract can't be rolled/settled without a known
  expiry) -- the surviving good rows on those same days are still used.
  None of the 8 Rule-B trigger or block-end dates land on one of these 164
  days; 28 of NSE's real expiry Thursdays (all in May, every year 2017-2024
  -- a systematic pattern in the archive, cause not independently
  diagnosable from this parsed file alone) do coincide with a degraded day,
  so the roll engine below counts and reports every time an actual roll in
  this simulation lands on one of those 28 dates, rather than assuming it
  doesn't matter.

SIZING: FLAT fractions only (the same CAPS sweep as v2), not severity-
scaled. v2 already tested severity-scaled sizing against flat, on these
exact same trigger dates, across four other variants, and flat won every
single time -- re-testing severity here on the same triggers would be
pure repetition of an already-settled comparison, not new information.

WHAT THIS DOES NOT DO: model whole-lot-size rounding (NIFTY option lot
sizes changed multiple times across 2017-2026 and this project has not
independently verified that full history) -- position sizes are continuous
fractional units, a disclosed idealization consistent with how every other
variant in this project has been sized. Real trading requires rounding to
whatever lot size was in force on that date, which matters more the
smaller the absolute premium spend is.

A SECOND, DIFFERENT DATA QUALITY ISSUE, found while debugging why the very
first (and largest) block only rolled 5 times instead of the ~22 a 473-day
monthly-rolling hold should produce: 8 distinct stretches of trading days
(every year's late-April/May, confirmed across all 2,403 pulled days) where
NSE's archive lists almost nothing beyond the about-to-expire month's
contracts -- 4-5 trading days most years, but 19 and 24 days in 2017 and
2018 respectively. A roll attempt landing in one of these gaps originally
just gave up for the REST of the block (freezing the position, silently
understating ~450 remaining days of the largest block). Fixed: a roll now
looks forward up to MAX_ROLL_LOOKAHEAD_DAYS (30, comfortably covering the
longest real gap found) for the first day the target contract actually has
data, earning 0% (idle cash) on the bridging days -- counted as n_gap_days
and reported per variant/cap, not hidden inside a flat multi-month freeze.
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import benchmark_signal as B
import hedge_fraction_sweep_v2 as H

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTIONS_FILE = os.path.join(HERE, "data", "pit", "nifty_options_daily.csv")

# ---- options-specific transaction costs, sourced -- see module docstring ----
OPT_BROKERAGE_FLAT = 20.0
OPT_STT_SELL_RATE = 0.0015        # sell-to-close a long put, OR write-to-open a call: on premium
OPT_STT_EXERCISE_RATE = 0.0015    # ITM cash settlement at expiry: on intrinsic value (separate base)
OPT_EXCH_RATE = 0.00035           # ~0.035% of premium, both legs
OPT_STAMP_RATE = 0.00003          # 0.003% of premium, buy side only
GST_RATE = 0.18
MIN_PREMIUM_VALUE = 1.0

# 28 real NSE expiry Thursdays (all in May, 2017-2024) that coincide with one
# of the 164 archive days where most of that day's EXPIRY_DT field is blank
# -- see module docstring. Populated once from the real data in load_option_chain().
DEGRADED_EXPIRY_DATES: set = set()

# NSE introduced WEEKLY Nifty expiries partway through this backtest window
# (only monthly existed in 2017-2019; weeklies dominate the near-term
# expiries from 2019 on). Picking the globally-nearest expiry would silently
# roll monthly pre-2019 and weekly post-2019 -- two very different decay/
# transaction-cost regimes -- without that being a deliberate choice. Instead
# every roll here always targets the standard MONTHLY expiry (last listed
# expiry in its calendar month), consistent across the whole 2017-2026
# window, and the more realistic choice for a multi-week Rule-B hold anyway
# (weeklies are a short-dated tactical instrument, not a crash-hedge one).
# Populated once from the real data in load_option_chain().
MONTHLY_EXPIRY_DATES: set = set()


def opt_buy_cost(premium_value: float) -> float:
    """Buy-to-open a long option, or buy-to-close a short one. STT-free."""
    if premium_value < MIN_PREMIUM_VALUE:
        return 0.0
    exch = premium_value * OPT_EXCH_RATE
    stamp = premium_value * OPT_STAMP_RATE
    gst = GST_RATE * (OPT_BROKERAGE_FLAT + exch)
    return OPT_BROKERAGE_FLAT + exch + stamp + gst


def opt_sell_cost(premium_value: float) -> float:
    """Sell-to-close a long option early, or write/sell-to-open a call."""
    if premium_value < MIN_PREMIUM_VALUE:
        return 0.0
    stt = premium_value * OPT_STT_SELL_RATE
    exch = premium_value * OPT_EXCH_RATE
    gst = GST_RATE * (OPT_BROKERAGE_FLAT + exch)
    return OPT_BROKERAGE_FLAT + stt + exch + gst


def opt_exercise_cost(intrinsic_value: float) -> float:
    """ITM cash settlement at expiry -- STT only, on intrinsic value (see
    docstring for the disclosed no-extra-brokerage simplification)."""
    if intrinsic_value < MIN_PREMIUM_VALUE:
        return 0.0
    return intrinsic_value * OPT_STT_EXERCISE_RATE


# ---------------------------------------------------------------- data loading
def load_option_chain():
    """Returns (price_lookup: dict[(date,expiry,strike,type)] -> close,
    by_date: dict[date] -> DataFrame of that day's good (non-null expiry)
    rows). Also populates DEGRADED_EXPIRY_DATES from the real data. Reads
    the source CSV exactly once."""
    raw = pd.read_csv(OPTIONS_FILE, parse_dates=["trade_date", "expiry_date"])
    n_total = len(raw)
    bad_mask = raw["expiry_date"].isna()
    n_bad = int(bad_mask.sum())
    bad_trade_dates = set(raw.loc[bad_mask, "trade_date"].unique())

    df = raw[~bad_mask].copy()
    df = df.drop_duplicates(subset=["trade_date", "expiry_date", "strike", "option_type"], keep="last")
    print(f"  [options data] {n_total:,} rows loaded, {n_bad:,} ({n_bad/n_total*100:.1f}%) dropped "
         f"(blank expiry_date in NSE's own archive -- see script docstring), "
         f"{len(df):,} usable rows remain")

    global DEGRADED_EXPIRY_DATES, MONTHLY_EXPIRY_DATES
    all_expiries = set(df["expiry_date"].unique())
    DEGRADED_EXPIRY_DATES = all_expiries & bad_trade_dates
    print(f"  [options data] {len(DEGRADED_EXPIRY_DATES)} real expiry dates coincide with a degraded "
         f"archive day -- rolls landing on these are counted separately in the report, not hidden")

    expiry_series = pd.Series(sorted(all_expiries))
    MONTHLY_EXPIRY_DATES = set(expiry_series.groupby(expiry_series.dt.to_period("M")).max())
    print(f"  [options data] {len(MONTHLY_EXPIRY_DATES)} of {len(all_expiries)} listed expiries are the "
         f"standard monthly expiry -- only these are used for rolling (see script docstring on weeklies)")

    price_lookup = df.set_index(["trade_date", "expiry_date", "strike", "option_type"])["close"].to_dict()
    by_date = {d: g for d, g in df.groupby("trade_date")}
    return price_lookup, by_date


def select_contract(date, by_date, option_type: str, spot: float, min_expiry=None, otm_pct: float = 0.0):
    """Picks the nearest-expiry (or nearest expiry STRICTLY AFTER min_expiry,
    when rolling), closest-to-target-strike, LIQUID (volume>0 that day)
    contract available on `date`. Falls back to any listed strike at that
    expiry if nothing traded that specific day (flagged, not silent)."""
    day_df = by_date.get(date)
    if day_df is None or not np.isfinite(spot):
        return None
    sub = day_df[(day_df["option_type"] == option_type) & (day_df["expiry_date"].isin(MONTHLY_EXPIRY_DATES))]
    sub = sub[sub["expiry_date"] > min_expiry] if min_expiry is not None else sub[sub["expiry_date"] >= date]
    if sub.empty:
        return None
    nearest_expiry = sub["expiry_date"].min()
    chain = sub[sub["expiry_date"] == nearest_expiry].copy()

    target_strike = spot * (1.0 - otm_pct) if option_type == "PE" else spot * (1.0 + otm_pct)
    chain["dist"] = (chain["strike"] - target_strike).abs()

    liquid = chain[chain["volume"] > 0]
    pool = liquid if not liquid.empty else chain
    illiquid_fallback = liquid.empty

    row = pool.sort_values("dist").iloc[0]
    return dict(expiry=nearest_expiry, strike=float(row["strike"]), entry_price=float(row["close"]),
               illiquid_fallback=bool(illiquid_fallback))


# A REAL, RECURRING NSE archive gap: every year's data shows a stretch (~5
# trading days most years, but 19 and 24 days in 2017 and 2018 respectively)
# around late April/May where the archive lists almost nothing beyond the
# about-to-expire month -- confirmed across all 2,403 pulled days, 8 distinct
# episodes, none longer than 24 trading days. Rather than treat a roll
# attempt landing in one of these gaps as "can't hedge for the rest of the
# block" (which, for the single longest block in this backtest, would have
# silently frozen the position for ~450 remaining days after a 5-day gap),
# a roll looks forward up to this many trading days for the first day the
# target contract actually has data, and rolls there instead -- the
# position earns 0% for those specific gap days (disclosed, not hidden;
# counted in n_gap_days below), which is a far smaller idealization than
# abandoning the hedge for the rest of the hold window.
MAX_ROLL_LOOKAHEAD_DAYS = 30


def find_roll_contract(dates_forward, by_date, option_type: str, spot: pd.Series,
                       min_expiry, otm_pct: float):
    """Tries select_contract on dates_forward[0], then successive days up to
    MAX_ROLL_LOOKAHEAD_DAYS ahead, for the archive-gap case above. Returns
    (contract_or_None, roll_date_used, n_gap_days)."""
    for gap, d in enumerate(dates_forward[:MAX_ROLL_LOOKAHEAD_DAYS]):
        s = float(spot.get(d, np.nan))
        c = select_contract(d, by_date, option_type, s, min_expiry=min_expiry, otm_pct=otm_pct)
        if c is not None:
            return c, d, gap
    return None, None, MAX_ROLL_LOOKAHEAD_DAYS


# ---------------------------------------------------------------- long put engine
def run_put_block(slice_value_at_entry: float, block_dates: pd.DatetimeIndex, spot: pd.Series,
                  by_date: dict, price_lookup: dict, otm_pct: float) -> dict:
    """Simulates ONE full long-put hedge cycle across block_dates, including
    every roll at expiry. Returns a dict with daily_values (np.ndarray, same
    length as block_dates -- the slice's rupee value each day, post all
    costs realized so far), total_costs, n_rolls, n_illiquid_fallbacks,
    n_degraded_day_rolls, opened (whether a contract could be opened at
    all)."""
    n_days = len(block_dates)
    daily_values = np.empty(n_days)
    total_costs = 0.0
    n_rolls = 0
    n_illiquid = 0
    n_degraded = 0
    n_gap_days = 0
    option_type = "PE"

    entry_date = block_dates[0]
    s0 = float(spot.get(entry_date, np.nan))
    contract = select_contract(entry_date, by_date, option_type, s0, otm_pct=otm_pct)
    if contract is None:
        daily_values[:] = slice_value_at_entry
        return dict(daily_values=daily_values, total_costs=0.0, n_rolls=0, n_illiquid=0,
                   n_degraded=0, n_gap_days=0, opened=False)

    entry_cost = opt_buy_cost(slice_value_at_entry)
    total_costs += entry_cost
    cash = slice_value_at_entry - entry_cost
    n_illiquid += int(contract["illiquid_fallback"])
    units = cash / contract["entry_price"] if contract["entry_price"] > 0 else 0.0
    current_expiry, current_strike = contract["expiry"], contract["strike"]
    position_value = units * contract["entry_price"]
    daily_values[0] = position_value

    k = 1
    while k < n_days:
        d = block_dates[k]
        s = float(spot.get(d, np.nan))

        if d < current_expiry:
            px = price_lookup.get((d, current_expiry, current_strike, option_type))
            if px is None:
                n_illiquid += 1
                px = position_value / units if units > 0 else 0.0
            position_value = units * px
            daily_values[k] = position_value
            k += 1
            continue

        intrinsic = max(current_strike - s, 0.0) if np.isfinite(s) else 0.0
        realized = units * intrinsic
        if intrinsic > 0:
            ex_cost = opt_exercise_cost(realized)
            total_costs += ex_cost
            realized -= ex_cost
        position_value = realized
        if current_expiry in DEGRADED_EXPIRY_DATES:
            n_degraded += 1
        daily_values[k] = position_value

        if k == n_days - 1:
            break  # block ends exactly on expiry day -- nothing left to roll into

        new_contract, roll_date, gap = find_roll_contract(block_dates[k:], by_date, option_type, spot,
                                                          current_expiry, otm_pct)
        n_rolls += 1
        n_gap_days += gap
        if new_contract is None:
            daily_values[k + 1:] = position_value
            return dict(daily_values=daily_values, total_costs=total_costs, n_rolls=n_rolls,
                       n_illiquid=n_illiquid, n_degraded=n_degraded, n_gap_days=n_gap_days, opened=True)

        # fill any gap days between expiry and the day the roll actually
        # executes with the FLAT (idle cash, 0% return) position value --
        # disclosed in the module docstring, not silent
        gap_end_k = k + gap
        if gap > 0:
            daily_values[k + 1:gap_end_k + 1] = position_value

        n_illiquid += int(new_contract["illiquid_fallback"])
        buy_c = opt_buy_cost(position_value)
        total_costs += buy_c
        cash2 = position_value - buy_c
        units = cash2 / new_contract["entry_price"] if new_contract["entry_price"] > 0 else 0.0
        current_expiry, current_strike = new_contract["expiry"], new_contract["strike"]
        position_value = units * new_contract["entry_price"]
        daily_values[gap_end_k] = position_value
        k = gap_end_k + 1

    last_date = block_dates[-1]
    if last_date < current_expiry:
        close_cost = opt_sell_cost(position_value)
        total_costs += close_cost
        position_value -= close_cost
        daily_values[-1] = position_value

    return dict(daily_values=daily_values, total_costs=total_costs, n_rolls=n_rolls,
               n_illiquid=n_illiquid, n_degraded=n_degraded, n_gap_days=n_gap_days, opened=True)


def simulate_put_hedge(nifty_ret: pd.Series, out_mask: pd.Series, spot: pd.Series, by_date: dict,
                       price_lookup: dict, otm_pct: float, flat_fraction: float,
                       initial_capital: float) -> dict:
    """Outer block-scoped loop, structurally identical to H.simulate()'s core-
    slice mechanics (equity sale + tax on the slice, unwind buy-back) but
    with the hedge leg computed by run_put_block (handles its own rolls)
    instead of a precomputed per-day return series -- options can't be
    reduced to one global return series because the contract held depends
    on which block/entry produced it."""
    idx = out_mask.index
    n = len(idx)
    blocks = H.find_hold_blocks(out_mask)
    H.build_cost_basis_cache(nifty_ret)

    portfolio = initial_capital
    equity = np.empty(n)
    total_costs = 0.0
    total_tax = 0.0
    n_cycles_hedged = 0
    n_cycles_skipped = 0
    n_rolls_total = 0
    n_illiquid_total = 0
    n_degraded_total = 0
    n_gap_days_total = 0

    i = 0
    while i < n:
        started_block = False
        for s, e in blocks:
            if s == i:
                started_block = True
                if flat_fraction > 0 and portfolio * flat_fraction >= H.MIN_TRADE_VALUE:
                    slice_value = portfolio * flat_fraction
                    core_value = portfolio - slice_value
                    cost = H.sell_cost(slice_value)
                    slice_basis = H.portfolio_cost_basis_at(i, flat_fraction, initial_capital)
                    gain = max(0.0, slice_value - slice_basis)
                    tax = gain * H.BLENDED_TAX_RATE
                    slice_value -= (tax + cost)
                    total_tax += tax
                    total_costs += cost

                    block_dates = idx[i:e + 1]
                    result = run_put_block(slice_value, block_dates, spot, by_date, price_lookup, otm_pct)
                    total_costs += result["total_costs"]
                    n_rolls_total += result["n_rolls"]
                    n_illiquid_total += result["n_illiquid"]
                    n_degraded_total += result["n_degraded"]
                    n_gap_days_total += result["n_gap_days"]
                    if result["opened"]:
                        n_cycles_hedged += 1
                    else:
                        n_cycles_skipped += 1

                    final_slice_value = float(result["daily_values"][-1])
                    hedge_gain = max(0.0, final_slice_value - slice_value)
                    hedge_tax = hedge_gain * H.HEDGE_GAIN_TAX_RATE
                    final_slice_value -= hedge_tax
                    total_tax += hedge_tax

                    unwind_cost = H.buy_cost(final_slice_value)
                    final_slice_value -= unwind_cost
                    total_costs += unwind_cost

                    # rebuild the day-by-day portfolio path for this block: core
                    # leg compounds plainly, slice leg follows run_put_block's
                    # own daily path (both pre-unwind; unwind/tax hit only the
                    # LAST day, matching how the equity engine in v2 also only
                    # settles unwind cost/tax on the block's final day)
                    core_path = core_value * (1 + nifty_ret.iloc[i:e + 1]).cumprod().values
                    slice_path = result["daily_values"].copy()
                    for j, pos in enumerate(range(i, e + 1)):
                        equity[pos] = core_path[j] + slice_path[j]
                    equity[e] = core_value_final = core_path[-1] + final_slice_value
                    portfolio = equity[e]
                else:
                    for pos in range(i, e + 1):
                        portfolio *= (1 + nifty_ret.iloc[pos])
                        equity[pos] = portfolio
                    n_cycles_skipped += 1
                i = e + 1
                break
        if not started_block:
            portfolio *= (1 + nifty_ret.iloc[i])
            equity[i] = portfolio
            i += 1

    equity = pd.Series(equity, index=idx)
    years = n / 252.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return dict(param=flat_fraction, cagr=cagr, total_return=total_return, max_dd=max_dd, calmar=calmar,
               total_costs=total_costs, total_tax=total_tax, n_cycles=len(blocks),
               n_cycles_hedged=n_cycles_hedged, n_cycles_skipped=n_cycles_skipped,
               n_rolls=n_rolls_total, n_illiquid_fallbacks=n_illiquid_total,
               n_degraded_day_rolls=n_degraded_total, n_gap_days=n_gap_days_total,
               final_value=float(equity.iloc[-1]), equity_curve=equity)


# ---------------------------------------------------------------- covered call engine
def run_covered_call_block(notional: float, block_dates: pd.DatetimeIndex, spot: pd.Series,
                           by_date: dict, price_lookup: dict) -> dict:
    """Writes (sells) an ATM call against `notional` rupees of the UNSOLD
    core position, rolled at each expiry. Returns daily P&L ADJUSTMENTS
    (rupees, to be ADDED on top of the full nifty_ret compounding of the
    core position -- this overlay does not touch the underlying at all,
    unlike the put engine's carved-out slice) plus cost/roll diagnostics."""
    n_days = len(block_dates)
    daily_pnl = np.zeros(n_days)
    total_costs = 0.0
    n_rolls = 0
    n_illiquid = 0
    n_degraded = 0
    n_gap_days = 0
    option_type = "CE"

    entry_date = block_dates[0]
    s0 = float(spot.get(entry_date, np.nan))
    contract = select_contract(entry_date, by_date, option_type, s0, otm_pct=0.0)
    if contract is None:
        return dict(daily_pnl=daily_pnl, total_costs=0.0, n_rolls=0, n_illiquid=0,
                   n_degraded=0, n_gap_days=0, opened=False)

    units = notional / s0 if s0 and np.isfinite(s0) else 0.0
    premium_received = units * contract["entry_price"]
    write_cost = opt_sell_cost(premium_received)
    total_costs += write_cost
    # NOT "premium_received - write_cost" here: the premium is already fully
    # captured by the day-to-day MTM telescoping below, which starts its
    # comparison AT this entry price and ends at the settlement/close price
    # -- it collapses to exactly (entry_price - exit_price), i.e. premium
    # collected minus what's owed/paid at exit, with no separate line item
    # needed. Booking the premium a second time here double-counted it (a
    # real bug found via a hand-worked example: sell @10, settle ITM @15
    # should net -5, the buggy version produced +5 -- fixed by only
    # charging the transaction cost at entry, not re-adding the premium).
    daily_pnl[0] -= write_cost
    n_illiquid += int(contract["illiquid_fallback"])
    current_expiry, current_strike = contract["expiry"], contract["strike"]
    prev_call_price = contract["entry_price"]

    k = 1
    while k < n_days:
        d = block_dates[k]
        s = float(spot.get(d, np.nan))

        if d < current_expiry:
            px = price_lookup.get((d, current_expiry, current_strike, option_type))
            if px is None:
                n_illiquid += 1
                px = prev_call_price
            # SHORT call: writer LOSES as the call's value rises
            daily_pnl[k] += -units * (px - prev_call_price)
            prev_call_price = px
            k += 1
            continue

        intrinsic = max(s - current_strike, 0.0) if np.isfinite(s) else 0.0
        settlement_pnl = -units * (intrinsic - prev_call_price)
        daily_pnl[k] += settlement_pnl
        if current_expiry in DEGRADED_EXPIRY_DATES:
            n_degraded += 1

        if k == n_days - 1:
            break  # block ends exactly on expiry day -- nothing left to roll into

        new_contract, roll_date, gap = find_roll_contract(block_dates[k:], by_date, option_type, spot,
                                                          current_expiry, 0.0)
        n_rolls += 1
        n_gap_days += gap
        if new_contract is None:
            break
        # gap days between expiry and the actual roll date earn no call-
        # writing P&L (no position held) -- daily_pnl there stays 0, same
        # disclosed idealization as the put engine's idle-cash gap days

        n_illiquid += int(new_contract["illiquid_fallback"])
        s_roll = float(spot.get(roll_date, np.nan))
        units = notional / s_roll if s_roll and np.isfinite(s_roll) else 0.0
        new_premium = units * new_contract["entry_price"]
        write_cost = opt_sell_cost(new_premium)
        total_costs += write_cost
        # same fix as the initial entry above -- cost only, premium is
        # captured by this new leg's own MTM telescoping from here on
        daily_pnl[k + gap] -= write_cost
        current_expiry, current_strike = new_contract["expiry"], new_contract["strike"]
        prev_call_price = new_contract["entry_price"]
        k = k + gap + 1

    last_date = block_dates[-1]
    if last_date < current_expiry:
        buy_back_cost = opt_buy_cost(units * prev_call_price)
        total_costs += buy_back_cost
        daily_pnl[-1] -= buy_back_cost

    return dict(daily_pnl=daily_pnl, total_costs=total_costs, n_rolls=n_rolls,
               n_illiquid=n_illiquid, n_degraded=n_degraded, n_gap_days=n_gap_days, opened=True)


def simulate_covered_call(nifty_ret: pd.Series, out_mask: pd.Series, spot: pd.Series, by_date: dict,
                          price_lookup: dict, flat_fraction: float, initial_capital: float) -> dict:
    """The core position is NEVER sold here (no equity tax event at all) --
    this overlay just adds the call-writing P&L on top of the full
    position's own nifty_ret compounding each day."""
    idx = out_mask.index
    n = len(idx)
    blocks = H.find_hold_blocks(out_mask)

    portfolio = initial_capital
    equity = np.empty(n)
    total_costs = 0.0
    total_tax = 0.0
    n_cycles_hedged = 0
    n_cycles_skipped = 0
    n_rolls_total = 0
    n_illiquid_total = 0
    n_degraded_total = 0
    n_gap_days_total = 0
    running_option_gain = 0.0  # taxed once per block at block end, same convention as the put engine

    i = 0
    while i < n:
        started_block = False
        for s, e in blocks:
            if s == i:
                started_block = True
                notional = portfolio * flat_fraction
                block_dates = idx[i:e + 1]
                if flat_fraction > 0 and notional >= H.MIN_TRADE_VALUE:
                    result = run_covered_call_block(notional, block_dates, spot, by_date, price_lookup)
                    total_costs += result["total_costs"]
                    n_rolls_total += result["n_rolls"]
                    n_illiquid_total += result["n_illiquid"]
                    n_degraded_total += result["n_degraded"]
                    n_gap_days_total += result["n_gap_days"]
                    if result["opened"]:
                        n_cycles_hedged += 1
                    else:
                        n_cycles_skipped += 1

                    block_option_gain = float(result["daily_pnl"].sum())
                    tax = max(0.0, block_option_gain) * H.HEDGE_GAIN_TAX_RATE
                    total_tax += tax
                    # tax deducted from the LAST day's P&L only, matching the
                    # put engine's convention of settling tax at block end
                    pnl = result["daily_pnl"].copy()
                    pnl[-1] -= tax

                    val = portfolio
                    for j, pos in enumerate(range(i, e + 1)):
                        val *= (1 + nifty_ret.iloc[pos])
                        val += pnl[j]
                        equity[pos] = val
                    portfolio = val
                else:
                    for pos in range(i, e + 1):
                        portfolio *= (1 + nifty_ret.iloc[pos])
                        equity[pos] = portfolio
                    n_cycles_skipped += 1
                i = e + 1
                break
        if not started_block:
            portfolio *= (1 + nifty_ret.iloc[i])
            equity[i] = portfolio
            i += 1

    equity = pd.Series(equity, index=idx)
    years = n / 252.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return dict(param=flat_fraction, cagr=cagr, total_return=total_return, max_dd=max_dd, calmar=calmar,
               total_costs=total_costs, total_tax=total_tax, n_cycles=len(blocks),
               n_cycles_hedged=n_cycles_hedged, n_cycles_skipped=n_cycles_skipped,
               n_rolls=n_rolls_total, n_illiquid_fallbacks=n_illiquid_total,
               n_degraded_day_rolls=n_degraded_total, n_gap_days=n_gap_days_total,
               final_value=float(equity.iloc[-1]), equity_curve=equity)


def main():
    print("Loading real NIFTY options chain...")
    price_lookup, by_date = load_option_chain()

    b = B.build_benchmark()
    px, oos_out_mask = b["px"], b["oos_out_mask"]
    idx = oos_out_mask.index
    nifty_ret = px.pct_change().reindex(idx).fillna(0.0)
    spot = px.reindex(idx)

    variants = {
        "hedge_put_atm": ("put", 0.0),
        "hedge_put_otm5": ("put", 0.05),
        "hedge_covered_call": ("call", None),
    }

    all_rows = []
    equity_curves = {}
    for variant_name, (kind, otm_pct) in variants.items():
        print(f"\nRunning {variant_name}...")
        for cap in H.CAPS:
            if kind == "put":
                row = simulate_put_hedge(nifty_ret, oos_out_mask, spot, by_date, price_lookup,
                                         otm_pct, cap, H.CURRENT_PORTFOLIO_VALUE)
            else:
                row = simulate_covered_call(nifty_ret, oos_out_mask, spot, by_date, price_lookup,
                                            cap, H.CURRENT_PORTFOLIO_VALUE)
            row["variant"] = variant_name
            row["mode"] = "flat"
            equity_curves[(variant_name, cap)] = row.pop("equity_curve")
            all_rows.append(row)
            print(f"  cap={cap:.2f}  CAGR={row['cagr']*100:6.2f}%  maxDD={row['max_dd']*100:7.2f}%  "
                 f"Calmar={row['calmar']:.3f}  costs=Rs{row['total_costs']:,.0f}  "
                 f"rolls={row['n_rolls']}  illiquid_fallbacks={row['n_illiquid_fallbacks']}  "
                 f"degraded_day_rolls={row['n_degraded_day_rolls']}  gap_days={row['n_gap_days']}  "
                 f"cycles_hedged={row['n_cycles_hedged']}/{row['n_cycles']}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "options_hedge_backtest.csv"), index=False)

    pd.set_option("display.width", 220)
    summary_rows = []
    for variant_name in variants:
        sub = df[df["variant"] == variant_name]
        if sub["calmar"].notna().sum() == 0:
            continue
        best = sub.loc[sub["calmar"].idxmax()]
        summary_rows.append(dict(variant=variant_name, best_param=best["param"], cagr=best["cagr"],
                                 max_dd=best["max_dd"], calmar=best["calmar"],
                                 total_costs=best["total_costs"], n_rolls=best["n_rolls"],
                                 final_value=best["final_value"]))
    summary = pd.DataFrame(summary_rows).sort_values("calmar", ascending=False)
    print("\n" + "=" * 110)
    print(f"BEST PER VARIANT (real options data), by Calmar -- Rs {H.CURRENT_PORTFOLIO_VALUE:,.0f} starting capital")
    print("=" * 110)
    print(summary.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("ROBUSTNESS: per-cycle real option-leg P&L, mean vs TRIMMED mean (drop 1 highest + 1 lowest of 8) "
         "vs median -- at each variant's own best cap")
    print("=" * 110)
    blocks = H.find_hold_blocks(oos_out_mask)
    for variant_name in variants:
        sub = df[df["variant"] == variant_name]
        if sub["calmar"].notna().sum() == 0:
            continue
        best_cap = float(sub.loc[sub["calmar"].idxmax(), "param"])
        curve = equity_curves[(variant_name, best_cap)]
        bh = (1 + nifty_ret).cumprod() * H.CURRENT_PORTFOLIO_VALUE
        cycle_rets = []
        for s, e in blocks:
            strat_mult = curve.iloc[e] / curve.iloc[s - 1] if s > 0 else curve.iloc[e] / H.CURRENT_PORTFOLIO_VALUE
            bh_mult = bh.iloc[e] / bh.iloc[s - 1] if s > 0 else bh.iloc[e] / H.CURRENT_PORTFOLIO_VALUE
            cycle_rets.append((strat_mult - bh_mult))  # hedge's edge over buy-and-hold, this cycle
        cycle_rets = np.array(cycle_rets)
        trimmed = np.sort(cycle_rets)[1:-1] if len(cycle_rets) >= 3 else cycle_rets
        print(f"  {variant_name:<20s} (cap={best_cap:.2f})  n={len(cycle_rets)}  "
             f"mean_edge_vs_bh={np.mean(cycle_rets)*100:+7.2f}%  trimmed={np.mean(trimmed)*100:+7.2f}%  "
             f"median={np.median(cycle_rets)*100:+7.2f}%  per-cycle: {[f'{r*100:+.1f}%' for r in cycle_rets]}")

    write_excel(df, summary)
    print(f"\nSaved: output/options_hedge_backtest.csv, output/options_hedge_backtest.xlsx")
    return df, summary


def write_excel(df: pd.DataFrame, summary: pd.DataFrame):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    best_fill = PatternFill("solid", fgColor="C6EFCE")

    ws0 = wb.active
    ws0.title = "Summary (best per variant)"
    ws0.append(["Variant", "Best flat fraction", "CAGR", "Max drawdown", "Calmar", "Total costs",
               "N rolls", "Final Rs"])
    for c in ws0[1]:
        c.font = header_font
        c.fill = header_fill
    for _, r in summary.iterrows():
        ws0.append([r["variant"], r["best_param"], round(r["cagr"] * 100, 2), round(r["max_dd"] * 100, 2),
                   round(r["calmar"], 3), round(r["total_costs"]), int(r["n_rolls"]), round(r["final_value"])])
    for col, w in zip("ABCDEFGH", [20, 16, 10, 14, 10, 14, 10, 14]):
        ws0.column_dimensions[col].width = w

    for variant_name, sub in df.groupby("variant"):
        ws = wb.create_sheet(variant_name[:31])
        ws.append(["Flat fraction", "CAGR", "Max drawdown", "Calmar", "Total costs", "Total tax",
                  "N cycles hedged/total", "N rolls", "Illiquid fallbacks", "Degraded-day rolls",
                  "Gap days (archive lookahead)", "Final Rs"])
        for c in ws[1]:
            c.font = header_font
            c.fill = header_fill
        sub = sub.reset_index(drop=True)
        for _, r in sub.iterrows():
            ws.append([r["param"], round(r["cagr"] * 100, 2), round(r["max_dd"] * 100, 2),
                      round(r["calmar"], 3) if pd.notna(r["calmar"]) else None,
                      round(r["total_costs"]), round(r["total_tax"]),
                      f"{int(r['n_cycles_hedged'])}/{int(r['n_cycles'])}", int(r["n_rolls"]),
                      int(r["n_illiquid_fallbacks"]), int(r["n_degraded_day_rolls"]),
                      int(r["n_gap_days"]), round(r["final_value"])])
        if sub["calmar"].notna().any():
            best_idx = int(sub["calmar"].idxmax())
            for col in range(1, 13):
                ws.cell(row=best_idx + 2, column=col).fill = best_fill
        for col, w in zip("ABCDEFGHIJKL", [12, 10, 14, 10, 14, 12, 18, 10, 16, 16, 20, 14]):
            ws.column_dimensions[col].width = w

    wb.save(os.path.join(OUTPUT_DIR, "options_hedge_backtest.xlsx"))


if __name__ == "__main__":
    main()
