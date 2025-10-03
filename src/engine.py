
import pandas as pd, numpy as np, io

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100/(1+rs))

def zscore(s: pd.Series, win: int = 60, minp: int = 20) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - m) / sd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    h,l,c = high, low, close
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()

def load_price_csv(csv_path_or_buffer, timeframe: str = "D", tz: str|None=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path_or_buffer)
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower: return lower[n]
        return None
    dc = pick('data','date','datetime','timestamp'); cc = pick('zamkniecie','close','adj close','adj_close','c')
    hc = pick('najwyzszy','high','h'); lc = pick('najnizszy','low','l'); vc = pick('wolumen','volume','wolumen [szt]','vol','wol','v')
    if dc is None or cc is None:
        raise ValueError("CSV musi mieć kolumny Data/Date oraz Zamkniecie/Close.")
    dt = pd.to_datetime(df[dc], errors='coerce')
    out = pd.DataFrame({
        'Data': dt,
        'Zamkniecie': pd.to_numeric(df[cc], errors='coerce'),
        'Najwyzszy': pd.to_numeric(df[hc], errors='coerce') if hc else pd.to_numeric(df[cc], errors='coerce')*1.001,
        'Najnizszy': pd.to_numeric(df[lc], errors='coerce') if lc else pd.to_numeric(df[cc], errors='coerce')*0.999,
        'Volume': pd.to_numeric(df[vc], errors='coerce') if vc else 0.0,
    }).dropna(subset=['Data','Zamkniecie']).sort_values('Data').set_index('Data')

    if timeframe.upper().startswith("H"):
        out = out.resample('H').agg({'Zamkniecie':'last','Najwyzszy':'max','Najnizszy':'min','Volume':'sum'}).dropna(subset=['Zamkniecie'])
    else:
        out = out.resample('D').agg({'Zamkniecie':'last','Najwyzszy':'max','Najnizszy':'min','Volume':'sum'}).dropna(subset=['Zamkniecie'])

    out['ret'] = out['Zamkniecie'].pct_change()
    out['EMA15']  = out['Zamkniecie'].ewm(span=15,  adjust=False).mean()
    out['EMA40']  = out['Zamkniecie'].ewm(span=40,  adjust=False).mean()
    out['EMA100'] = out['Zamkniecie'].ewm(span=100, adjust=False).mean()
    out['RSI14']  = rsi(out['Zamkniecie'], 14)
    chg = out["Zamkniecie"].diff().fillna(0.0)
    direction = np.sign(chg)
    out["OBV"] = (direction * out["Volume"]).fillna(0.0).cumsum()
    out["VolZ60"] = zscore(out["Volume"], 60, 20)
    out["OBV_slope5"] = out["OBV"].diff(5)
    out["vol20"] = out["ret"].rolling(20, min_periods=10).std()
    out["ATR14"] = atr(out["Najwyzszy"], out["Najnizszy"], out["Zamkniecie"], 14)
    out["regime"] = classify_regime_rolling(out)
    return out

def classify_regime_rolling(df: pd.DataFrame, trend_band: float = 0.004, vol_win: int = 126) -> pd.Series:
    px, ema100 = df["Zamkniecie"], df["EMA100"]
    rel = (px - ema100) / (ema100 + 1e-12)
    trend = np.where(rel > trend_band, "bull", np.where(rel < -trend_band, "bear", "side"))
    vol = df["vol20"].fillna(method="bfill")
    q1 = vol.rolling(vol_win, min_periods=30).quantile(0.33)
    q2 = vol.rolling(vol_win, min_periods=30).quantile(0.66)
    vol_band = np.where(vol <= q1, "lowvol", np.where(vol <= q2, "midvol", "highvol"))
    return pd.Series([f"{t}_{v}" for t,v in zip(trend, vol_band)], index=df.index, name="regime")

BASE_REGIME_PARAMS = {
    "bull_lowvol":  dict(rsi_buy_floor=60, rsi_sell_ceiling=50, min_hold=4, dd_stop=0.25, vol_conf="mild"),
    "bull_midvol":  dict(rsi_buy_floor=58, rsi_sell_ceiling=52, min_hold=4, dd_stop=0.25, vol_conf="mild"),
    "bull_highvol": dict(rsi_buy_floor=55, rsi_sell_ceiling=55, min_hold=3, dd_stop=0.30, vol_conf="strict"),
    "side_lowvol":  dict(rsi_buy_floor=62, rsi_sell_ceiling=48, min_hold=3, dd_stop=0.20, vol_conf="off"),
    "side_midvol":  dict(rsi_buy_floor=60, rsi_sell_ceiling=50, min_hold=3, dd_stop=0.22, vol_conf="off"),
    "side_highvol": dict(rsi_buy_floor=58, rsi_sell_ceiling=52, min_hold=2, dd_stop=0.25, vol_conf="mild"),
    "bear_lowvol":  dict(rsi_buy_floor=65, rsi_sell_ceiling=55, min_hold=3, dd_stop=0.20, vol_conf="mild"),
    "bear_midvol":  dict(rsi_buy_floor=67, rsi_sell_ceiling=57, min_hold=3, dd_stop=0.22, vol_conf="strict"),
    "bear_highvol": dict(rsi_buy_floor=70, rsi_sell_ceiling=60, min_hold=2, dd_stop=0.18, vol_conf="strict"),
}

def apply_overrides(base: dict, ov: dict) -> dict:
    import copy
    out = copy.deepcopy(base)
    for regime, params in out.items():
        if 'rsi_buy_floor' in ov: params['rsi_buy_floor'] = int(np.clip(params['rsi_buy_floor'] + ov['rsi_buy_floor'], 50, 75))
        if 'rsi_sell_ceiling' in ov: params['rsi_sell_ceiling'] = int(np.clip(params['rsi_sell_ceiling'] + ov['rsi_sell_ceiling'], 40, 70))
        if 'min_hold' in ov: params['min_hold'] = int(np.clip(params['min_hold'] + ov['min_hold'], 1, 12))
        if 'dd_stop' in ov: params['dd_stop'] = float(np.clip(params['dd_stop'] + ov['dd_stop'], 0.05, 0.50))
        if 'vol_shift' in ov:
            levels = ["off","mild","strict"]; i = max(0, min(2, levels.index(params['vol_conf']) + int(ov['vol_shift'])))
            params['vol_conf'] = levels[i]
    return out

def merge_sentiment(df: pd.DataFrame, sent: pd.DataFrame|None) -> pd.DataFrame:
    out = df.copy()
    if sent is None or sent.empty:
        out["sentiment"] = 0.5
        return out
    s = sent.copy()
    lc = {c.lower(): c for c in s.columns}
    dtc = lc.get("date") or lc.get("data")
    sc  = lc.get("sentiment") or lc.get("score") or lc.get("value")
    if dtc is None or sc is None:
        out["sentiment"] = 0.5; return out
    s["Date"] = pd.to_datetime(s[dtc], errors="coerce")
    s = s.dropna(subset=["Date"]).set_index("Date").sort_index()[[sc]].rename(columns={sc:"sentiment"})
    s = s.resample("H" if (out.index.freq=='H' or out.index.inferred_freq=='H') else "D").mean().ffill().clip(0,1)
    out = out.join(s, how="left")
    out["sentiment"] = out["sentiment"].fillna(method="ffill").fillna(0.5)
    return out

def _volume_gate(z: float, obv_slope: float, mode: str) -> bool:
    if mode=="off": return True
    if mode=="mild":
        return (z is not None and z>=0) or (obv_slope is not None and obv_slope>0)
    if mode=="strict":
        return (z is not None and z>0.5) and (obv_slope is not None and obv_slope>0)
    return True

def ema_core_signals_regime(df_bt: pd.DataFrame, regime_params: dict|None=None, sentiment_gate: bool=True,
                            transaction_cost_pips: float = 0.0, is_fx: bool=True,
                            sl_mult: float = 1.5, tp_mult: float = 2.5):
    if regime_params is None: regime_params = BASE_REGIME_PARAMS
    px, ema15, ema40, ema100, rsi14 = df_bt['Zamkniecie'], df_bt['EMA15'], df_bt['EMA40'], df_bt['EMA100'], df_bt['RSI14']
    reg = df_bt['regime'] if 'regime' in df_bt.columns else pd.Series("side_midvol", index=df_bt.index)
    volz = df_bt["VolZ60"] if "VolZ60" in df_bt.columns else pd.Series(0.0, index=df_bt.index)
    obv_s = df_bt["OBV_slope5"] if "OBV_slope5" in df_bt.columns else pd.Series(0.0, index=df_bt.index)
    sent  = df_bt["sentiment"] if "sentiment" in df_bt.columns else pd.Series(0.5, index=df_bt.index)
    atr14 = df_bt["ATR14"] if "ATR14" in df_bt.columns else pd.Series(np.nan, index=df_bt.index)

    pip = transaction_cost_pips if is_fx else 0.0
    position=False; entry_i=None; entry_p=None; sl=None; tp=None; signals=[]; trades=[]
    for i in range(1, len(df_bt)):
        d = df_bt.index[i]; price=float(px.iloc[i])
        rp = regime_params.get(reg.iloc[i], regime_params["side_midvol"])
        buy_base  = (ema15.iloc[i] > ema40.iloc[i]) and (price > float(ema100.iloc[i]))
        sell_base = (ema15.iloc[i] < ema40.iloc[i]) or  (price < float(ema100.iloc[i]))

        v_ok_buy  = _volume_gate(float(volz.iloc[i]) if not pd.isna(volz.iloc[i]) else None,
                                 float(obv_s.iloc[i]) if not pd.isna(obv_s.iloc[i]) else None, rp.get("vol_conf","off"))
        v_ok_sell = _volume_gate(float(volz.iloc[i]) if not pd.isna(volz.iloc[i]) else None,
                                 float(obv_s.iloc[i]) if not pd.isna(obv_s.iloc[i]) else None, rp.get("vol_conf","off"))
        if sentiment_gate:
            if sent.iloc[i] >= 0.6: v_ok_sell = False
            if sent.iloc[i] <= 0.4: v_ok_buy = False

        if not position:
            if buy_base and (rsi14.iloc[i] > rp["rsi_buy_floor"]) and v_ok_buy:
                position=True; entry_i=i; entry_p=price
                a = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else price*0.002
                sl = price - sl_mult*a
                tp = price + tp_mult*a
                signals.append({"date": d, "type":"BUY", "price": price, "note": f"EMA BUY ({reg.iloc[i]})", "SL": sl, "TP": tp})
        else:
            a = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else price*0.002
            dd = 1.0 - (price/entry_p) if entry_p else 0.0
            hit_sl = (price <= sl) if sl is not None else False
            hit_tp = (price >= tp) if tp is not None else False
            exit_reason=None
            if hit_sl: exit_reason="SL hit"
            elif hit_tp: exit_reason="TP hit"
            elif ((i - entry_i) >= rp["min_hold"]) and ((sell_base and (rsi14.iloc[i] < rp["rsi_sell_ceiling"]) and v_ok_sell) or (dd >= rp["dd_stop"])):
                exit_reason = "Rule exit" if dd<rp["dd_stop"] else "DD stop"
            if exit_reason:
                signals.append({"date": d, "type":"SELL", "price": price, "note": exit_reason, "SL": sl, "TP": tp})
                pnl_pct = (price/entry_p - 1.0)
                if pip>0: pnl_pct -= (2*pip) / max(entry_p, 1e-9)
                pnl = pnl_pct*100.0
                trades.append({"entry_date": df_bt.index[entry_i], "exit_date": d, "entry_price": float(entry_p), "exit_price": price,
                               "bars": int(i-entry_i), "pnl_usd": float(pnl), "exit_reason": exit_reason, "sl": sl, "tp": tp})
                position=False; entry_i=None; entry_p=None; sl=None; tp=None
    return pd.DataFrame(signals), pd.DataFrame(trades)

def bench_roi(df: pd.DataFrame, start: str) -> float:
    view = df.loc[pd.to_datetime(start):]
    v0 = float(view['Zamkniecie'].iloc[0]); v1 = float(view['Zamkniecie'].iloc[-1])
    return (v1/v0 - 1.0) * 100.0

def rolling_optimize_signals(df: pd.DataFrame, start="2022-01-01",
                             window_bars: int = 24*180, step_bars: int = 24*7,
                             n_samples: int = 200, seed: int = 13,
                             param_grid: dict|None = None, sentiment: pd.DataFrame|None = None,
                             transaction_cost_pips: float = 2.0, is_fx: bool=True,
                             sl_mult: float = 1.5, tp_mult: float = 2.5):
    rng = np.random.default_rng(seed)
    df_full = merge_sentiment(df, sentiment)
    df_bt = df_full.loc[pd.to_datetime(start):].copy()
    dates = df_bt.index
    if param_grid is None:
        param_grid = {"rsi_buy_floor": [-5,-3,0,+2,+4], "rsi_sell_ceiling": [-4,-2,0,+2,+4],
                      "min_hold": [-1,0,+1,+2], "dd_stop": [-0.05,-0.03,0.0,+0.03,+0.05], "vol_shift": [-1,0,+1]}
    all_signals=[]; all_trades=[]
    left = 0
    while True:
        ins_right_idx = min(left + window_bars, len(dates)-1)
        oos_right_idx = min(ins_right_idx + step_bars, len(dates)-1)
        ins = df_bt.iloc[left:ins_right_idx+1]
        oos = df_bt.iloc[ins_right_idx:oos_right_idx+1]
        if len(oos)<10: break
        best_score=-1e18; best_ov=None
        for _ in range(n_samples):
            ov = {"rsi_buy_floor": int(rng.choice(param_grid["rsi_buy_floor"])),
                  "rsi_sell_ceiling": int(rng.choice(param_grid["rsi_sell_ceiling"])),
                  "min_hold": int(rng.choice(param_grid["min_hold"])),
                  "dd_stop": float(rng.choice(param_grid["dd_stop"])),
                  "vol_shift": int(rng.choice(param_grid["vol_shift"]))}
            rp = apply_overrides(BASE_REGIME_PARAMS, ov)
            sig,trd = ema_core_signals_regime(ins, regime_params=rp, sentiment_gate=True,
                                              transaction_cost_pips=transaction_cost_pips, is_fx=is_fx,
                                              sl_mult=sl_mult, tp_mult=tp_mult)
            roi = float(trd['pnl_usd'].sum()) if not trd.empty else -999.0
            trades=len(trd); score = roi + 0.15*trades
            if score>best_score: best_score=score; best_ov=ov
        rp_best = apply_overrides(BASE_REGIME_PARAMS, best_ov or {})
        sig_oos, trd_oos = ema_core_signals_regime(oos, regime_params=rp_best, sentiment_gate=True,
                                                   transaction_cost_pips=transaction_cost_pips, is_fx=is_fx,
                                                   sl_mult=sl_mult, tp_mult=tp_mult)
        all_signals += sig_oos.to_dict("records"); all_trades  += trd_oos.to_dict("records")
        left = oos_right_idx
        if left >= len(dates)-10: break
    sig_df = pd.DataFrame(all_signals); trd_df = pd.DataFrame(all_trades)
    bh = bench_roi(df_full, start)
    roi = float(trd_df['pnl_usd'].sum()) if not trd_df.empty else -999.0
    info={"mode":"rolling_model" if roi>=bh else "fallback_bh","bench_roi_pct": bh,
          "rolling":{"window_bars":window_bars,"step_bars":step_bars,"n_samples":n_samples},
          "sl_mult": sl_mult, "tp_mult": tp_mult, "tx_pips": transaction_cost_pips}
    if roi < bh:
        sig_df = pd.DataFrame([{"date": df_bt.index[0], "type":"BUY", "price": float(df_bt['Zamkniecie'].iloc[0]), "note":"B&H fallback"}])
        trd_df = pd.DataFrame([{"entry_date": df_bt.index[0], "exit_date": df_bt.index[-1],
                                "entry_price": float(df_bt['Zamkniecie'].iloc[0]), "exit_price": float(df_bt['Zamkniecie'].iloc[-1]),
                                "bars": len(df_bt)-1, "pnl_usd": float(bh), "exit_reason":"fallback"}])
    return info, sig_df, trd_df

def heatmap_sweep(df: pd.DataFrame, start="2022-01-01",
                  rsi_buy_floor_vals=(55,58,60,62,65), min_hold_vals=(2,3,4,5),
                  dd_stop_vals=(0.15,0.20,0.25,0.30), transaction_cost_pips: float = 2.0, is_fx: bool=True):
    import itertools
    rows=[]
    df_bt = df.loc[pd.to_datetime(start):]
    for rsi_b, mh in itertools.product(rsi_buy_floor_vals, min_hold_vals):
        roi_acc=0.0; cnt=0
        for dd in dd_stop_vals:
            rp = apply_overrides(BASE_REGIME_PARAMS, {"rsi_buy_floor": rsi_b-60, "min_hold": mh-4, "dd_stop": dd-0.25})
            sig,trd = ema_core_signals_regime(df_bt, regime_params=rp, sentiment_gate=True,
                                              transaction_cost_pips=transaction_cost_pips, is_fx=is_fx)
            roi = float(trd['pnl_usd'].sum()) if not trd.empty else -999.0
            roi_acc += roi; cnt += 1
        rows.append({"rsi_buy_floor": rsi_b, "min_hold": mh, "roi": roi_acc/max(cnt,1)})
    hm = pd.DataFrame(rows).pivot(index="rsi_buy_floor", columns="min_hold", values="roi")
    return hm
