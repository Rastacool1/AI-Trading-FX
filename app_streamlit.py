import io, sys, requests
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go, plotly.express as px
import streamlit as st
ROOT=Path(__file__).parent; SRC=ROOT/'src'; sys.path.append(str(SRC))
from engine import load_price_csv, rolling_optimize_signals, heatmap_sweep, quick_autotune
st.set_page_config(page_title='AI Trader — v4.1', layout='wide')
st.title('AI Trader — v4.1 • Auto‑Tune (RSI/SL/TP) + H1/D1 + Heatmap')
@st.cache_data(show_spinner=False)
def fetch_stooq(s,interval='d'):
 url=f'https://stooq.pl/q/d/l/?s={s}&i={interval}'; r=requests.get(url,timeout=10); return r.text if r.status_code==200 and r.text.strip() else None
def parse_df(txt,tf):
 return load_price_csv(io.StringIO(txt), timeframe=('H' if 'H1' in tf else 'D')) if txt else None
def plot_equity(df,start,trd,title):
 idx=df.loc[pd.to_datetime(start):].index; curve=pd.Series(100.0,index=idx)
 if trd is not None and len(trd)>0:
  cum=100.0
  for _,t in trd.iterrows():
   d=pd.to_datetime(t['exit_date']) if 'exit_date' in t else None
   if d is not None and d in curve.index:
    cum+=float(t['pnl_usd']); curve.loc[d:]=cum
 bh=100.0*(df.loc[pd.to_datetime(start):,'Zamkniecie']/df.loc[pd.to_datetime(start):,'Zamkniecie'].iloc[0])
 fig=go.Figure(); fig.add_trace(go.Scatter(x=idx,y=curve,name='Model')); fig.add_trace(go.Scatter(x=idx,y=bh,name='Buy&Hold'))
 return fig.update_layout(height=520,title=title,xaxis=dict(rangeslider=dict(visible=True),type='date'))
with st.sidebar:
 st.header('Dane'); src=st.radio('Źródło',['Wgraj CSV','Stooq'],index=0); tf=st.selectbox('Interwał docelowy',['D1 (dzień)','H1 (godzina)'],index=1); start=st.text_input('Start','2022-01-01');
 st.header('Rolling'); window=st.number_input('Okno (bary)',value=4320,step=168); step=st.number_input('Krok (bary)',value=168,step=24); samples=st.slider('Próby/okno',50,400,200,25); seed=st.number_input('Seed',13)
 st.header('FX & koszty'); is_fx=st.checkbox('Para FX',True); tx=st.number_input('Koszt (pips)',2.0,step=0.5)
 st.header('SL/TP'); sl=st.number_input('SL=ATR×',1.5,step=0.1); tp=st.number_input('TP=ATR×',2.5,step=0.1)
 auto=st.button('🔧 Auto‑Tune'); noreg=st.checkbox('No‑regret',True); run=st.button('▶️ Przelicz SINGLE'); heat=st.button('🗺️ Heatmap')
if 'csv' not in st.session_state: st.session_state.csv=None
if 'sug' not in st.session_state: st.session_state.sug={}
df=None
if src=='Wgraj CSV':
 up=st.file_uploader('CSV',type=['csv']);
 if up is not None: st.session_state.csv=up.read().decode('utf-8','ignore'); st.success('CSV zapisany w sesji.')
else:
 sym=st.text_input('Ticker','eurusd'); interval=st.selectbox('Interwał',['d','w','m'],0)
 if st.button('Pobierz'):
  t=fetch_stooq(sym,interval); st.session_state.csv=t if t else None; st.success('Pobrano w sesji.') if t else st.error('Brak danych')
if st.session_state.csv:
 try:
  df=parse_df(st.session_state.csv,tf)
 except Exception as e:
  st.error(f'Problem z CSV: {e}')
tabs=st.tabs(['📈 Przegląd','🧾 Sygnały','🗺️ Heatmap'])
if auto:
 if df is None or len(df)<300: st.warning('Za mało danych do Auto‑Tune (min. 300 barów).')
 else:
  st.session_state.sug=quick_autotune(df,start,is_fx,tx); st.success(str(st.session_state.sug))
if run:
 if df is None or len(df)<100: st.warning('Brak danych lub zbyt mało barów.')
 else:
  slx=float(st.session_state.sug.get('sl_mult',sl)); tpx=float(st.session_state.sug.get('tp_mult',tp))
  info,sig,trd=rolling_optimize_signals(df,start,window,step,int(samples),int(seed),tx,is_fx,slx,tpx, bool(noreg))
  with tabs[0]: st.plotly_chart(plot_equity(df,start,trd,f'Equity ({tf})'), use_container_width=True)
  with tabs[1]: st.dataframe(sig, use_container_width=True, height=420)
if heat and df is not None and len(df)>=100:
 with tabs[2]: st.write('Heatmap demo');
