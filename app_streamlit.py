
import io, sys, requests
from pathlib import Path
import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path: sys.path.append(str(SRC))
from engine import load_price_csv, rolling_optimize_signals, heatmap_sweep

st.set_page_config(page_title="AI Trader — v4.0.1", layout="wide")
st.title("AI Trader — v4.0.1 • FX Hourly + SL/TP + Heatmap")

@st.cache_data(show_spinner=False)
def fetch_stooq(symbol: str, interval: str = "d") -> str | None:
    url = f"https://stooq.pl/q/d/l/?s={symbol}&i={interval}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200 or len(r.text.strip())==0:
        return None
    return r.text

def parse_df_from_csv_text(csv_text: str, timeframe_label: str):
    if not csv_text:
        return None
    return load_price_csv(io.StringIO(csv_text), timeframe=("H" if timeframe_label=="H1" else "D"))

def plot_equity(df: pd.DataFrame, start: str, trd: pd.DataFrame, title: str):
    idx = df.loc[pd.to_datetime(start):].index
    curve = pd.Series(100.0, index=idx, name="Model")
    if trd is not None and not trd.empty:
        cum=100.0
        for _, t in trd.iterrows():
            d = pd.to_datetime(t['exit_date'])
            if d in curve.index:
                cum += float(t['pnl_usd']); curve.loc[d:] = cum
    bh = 100.0 * (df.loc[pd.to_datetime(start):,'Zamkniecie'] / df.loc[pd.to_datetime(start):,'Zamkniecie'].iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=curve, mode="lines", name="Model"))
    fig.add_trace(go.Scatter(x=idx, y=bh, mode="lines", name="Buy&Hold"))
    if trd is not None and not trd.empty:
        try:
            buys = [pd.to_datetime(x) for x in trd['entry_date']]
            sells= [pd.to_datetime(x) for x in trd['exit_date']]
            by = [curve.loc[d] if d in curve.index else None for d in buys]
            sy = [curve.loc[d] if d in curve.index else None for d in sells]
            fig.add_trace(go.Scatter(x=buys, y=by, mode="markers", marker_symbol="triangle-up", marker_size=10, name="BUY"))
            fig.add_trace(go.Scatter(x=sells,y=sy, mode="markers", marker_symbol="triangle-down", marker_size=10, name="SELL"))
        except Exception:
            pass
    fig.update_layout(height=520, title=title, xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return fig

# Sidebar
with st.sidebar:
    st.header("Dane")
    src_mode = st.radio("Źródło", ["Wgraj CSV", "Stooq"], index=0)
    timeframe = st.selectbox("Interwał docelowy", ["D1","H1"], index=1)
    start = st.text_input("Start (YYYY-MM-DD)", "2022-01-01")
    st.markdown('---')
    st.header("Rolling (bary)")
    window_bars = st.number_input("Okno (bary)", value=24*180, step=24*7)
    step_bars   = st.number_input("Krok (bary)", value=24*7, step=24)
    n_samples   = st.slider("Próby/okno", 50, 400, 200, step=25)
    seed        = st.number_input("Seed", value=13, step=1)
    st.markdown('---')
    st.header("FX & koszty")
    is_fx = st.checkbox("Para FX", value=True)
    tx_pips = st.number_input("Transaction cost (pips round-trip)", value=2.0, step=0.5)
    st.markdown('---')
    st.header("SL/TP (ATR)")
    sl_mult = st.number_input("SL = ATR ×", value=1.5, step=0.1)
    tp_mult = st.number_input("TP = ATR ×", value=2.5, step=0.1)
    st.markdown('---')
    go_single = st.button("Przelicz SINGLE", use_container_width=True)
    go_heat   = st.button("Pokaż Heatmap", use_container_width=True)

# Session state for persistence
if "csv_text" not in st.session_state:
    st.session_state.csv_text = None

df = None
if src_mode == "Wgraj CSV":
    up = st.file_uploader("CSV (EUR/USD H1 lub D1)", type=["csv"], key="upl")
    if up is not None:
        text = up.read().decode("utf-8", errors="ignore")
        st.session_state.csv_text = text
        st.success("CSV wczytany (zapisany w sesji).")
else:
    sym = st.text_input("Ticker Stooq (np. eurusd)", "eurusd", key="sym")
    interval = st.selectbox("Interwał Stooq", ["d","w","m"], index=0, key="intr")
    if st.button("Pobierz", key="btn_fetch"):
        text = fetch_stooq(sym.strip(), interval=interval)
        if text is None:
            st.error("Brak danych ze Stooq.")
        else:
            st.session_state.csv_text = text
            st.success("Pobrano i zapisano dane w sesji.")

# Build df if saved text exists
if st.session_state.csv_text:
    try:
        df = parse_df_from_csv_text(st.session_state.csv_text, timeframe)
    except Exception as e:
        st.error(f"Problem z parsowaniem CSV: {e}")

tabs = st.tabs(["📈 Przegląd", "🧾 Sygnały", "🗺️ Heatmap"])

if go_single:
    if df is None or len(df) < 100:
        st.warning("Brak danych lub za mało wierszy. Najpierw **Pobierz** / **Wgraj** CSV (min. 100 barów).")
    else:
        with st.spinner("Liczę rolling + SL/TP..."):
            info, sig, trd = rolling_optimize_signals(
                df, start=start, window_bars=int(window_bars), step_bars=int(step_bars),
                n_samples=int(n_samples), seed=int(seed),
                transaction_cost_pips=float(tx_pips), is_fx=bool(is_fx),
                sl_mult=float(sl_mult), tp_mult=float(tp_mult)
            )
        with tabs[0]:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Tryb", info['mode']); c2.metric("B&H ROI %", f"{info['bench_roi_pct']:.2f}")
            c3.metric("Transakcji", len(trd)); c4.metric("SL/TP", f"{info['sl_mult']}×ATR / {info['tp_mult']}×ATR")
            st.plotly_chart(plot_equity(df, start, trd, f"Equity ({timeframe})"), use_container_width=True)
        with tabs[1]:
            st.dataframe(sig, use_container_width=True, height=420)

if go_heat:
    if df is None or len(df) < 100:
        st.warning("Brak danych lub za mało wierszy do heatmapy. Najpierw **Pobierz** / **Wgraj** CSV.")
    else:
        with st.spinner("Liczę heatmapę..."):
            hm = heatmap_sweep(df, start=start, is_fx=bool(is_fx), transaction_cost_pips=float(tx_pips))
        with tabs[2]:
            st.subheader("ROI (średnia po dd_stop) vs RSI_buy × min_hold")
            fig = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
