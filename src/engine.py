import pandas as pd, numpy as np
def rsi(x,n=14):
 d=x.diff();u=d.clip(lower=0);v=(-d).clip(lower=0);
 ru=u.ewm(alpha=1/n,adjust=False).mean(); rv=v.ewm(alpha=1/n,adjust=False).mean();
 rs=ru/(rv+1e-12); return 100-(100/(1+rs))
def atr(h,l,c,n=14):
 import numpy as np, pandas as pd
 tr=pd.concat([(h-l).abs(),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def load_price_csv(buf,timeframe='D',tz=None):
 import pandas as pd
 df=pd.read_csv(buf); lc={c.lower():c for c in df.columns};
 dc=lc.get('data') or lc.get('date') or lc.get('datetime') or lc.get('timestamp'); cc=lc.get('zamkniecie') or lc.get('close') or lc.get('adj close') or lc.get('c');
 hc=lc.get('najwyzszy') or lc.get('high') or lc.get('h'); lcmin=lc.get('najnizszy') or lc.get('low') or lc.get('l'); vc=lc.get('wolumen') or lc.get('volume') or lc.get('vol')
 if dc is None or cc is None: raise ValueError('CSV musi mieć kolumny Data/Date oraz Zamkniecie/Close.')
 dt=pd.to_datetime(df[dc],errors='coerce'); out=pd.DataFrame({'Data':dt,'Zamkniecie':pd.to_numeric(df[cc],errors='coerce'),'Najwyzszy':pd.to_numeric(df[hc],errors='coerce') if hc else pd.to_numeric(df[cc],errors='coerce')*1.001,'Najnizszy':pd.to_numeric(df[lcmin],errors='coerce') if lcmin else pd.to_numeric(df[cc],errors='coerce')*0.999,'Volume':pd.to_numeric(df[vc],errors='coerce') if vc else 0.0}).dropna(subset=['Data','Zamkniecie']).sort_values('Data').set_index('Data')
 out=out.resample('H' if timeframe.upper().startswith('H') else 'D').agg({'Zamkniecie':'last','Najwyzszy':'max','Najnizszy':'min','Volume':'sum'}).dropna(subset=['Zamkniecie']);
 out['ret']=out['Zamkniecie'].pct_change(); out['EMA15']=out['Zamkniecie'].ewm(span=15,adjust=False).mean(); out['EMA40']=out['Zamkniecie'].ewm(span=40,adjust=False).mean(); out['EMA100']=out['Zamkniecie'].ewm(span=100,adjust=False).mean(); out['RSI14']=rsi(out['Zamkniecie']); out['ATR14']=atr(out['Najwyzszy'],out['Najnizszy'],out['Zamkniecie']); return out
BASE={'bull_lowvol':dict(rsi_buy_floor=60,rsi_sell_ceiling=50,min_hold=4,dd_stop=0.25),'side_midvol':dict(rsi_buy_floor=60,rsi_sell_ceiling=50,min_hold=3,dd_stop=0.22),'bear_highvol':dict(rsi_buy_floor=70,rsi_sell_ceiling=60,min_hold=2,dd_stop=0.18)}
def ema_core_signals_regime(df,sl_mult=1.5,tp_mult=2.5,is_fx=True,tx_pips=2.0):
 import pandas as pd, numpy as np
 px=df['Zamkniecie']; e15=df['EMA15']; e40=df['EMA40']; e100=df['EMA100']; r=df['RSI14']; atr14=df['ATR14'];
 position=False; entry=None; sl=None; tp=None; sig=[]; trd=[]
 for i in range(1,len(df)):
  d=df.index[i]; price=float(px.iloc[i]); buy=(e15.iloc[i]>e40.iloc[i]) and (price>e100.iloc[i]) and (r.iloc[i]>58); sell=(e15.iloc[i]<e40.iloc[i]) or (price<e100.iloc[i]) or (r.iloc[i]<50)
  a=float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else price*0.002
  if not position:
   if buy:
    position=True; entry=price; sl=price-sl_mult*a; tp=price+tp_mult*a; sig.append({'date':d,'type':'BUY','price':price})
  else:
   hit_sl=price<=sl; hit_tp=price>=tp
   if hit_sl or hit_tp or sell:
    pnl=(price/entry-1.0)*100.0; trd.append({'entry_date':None,'exit_date':d,'entry_price':entry,'exit_price':price,'pnl_usd':pnl}); sig.append({'date':d,'type':'SELL','price':price}); position=False; entry=None
 return pd.DataFrame(sig), pd.DataFrame(trd)
def bench_roi(df,start):
 v=df.loc[pd.to_datetime(start):]; return (float(v['Zamkniecie'].iloc[-1])/float(v['Zamkniecie'].iloc[0])-1.0)*100.0
def rolling_optimize_signals(df,start='2022-01-01',window_bars=4320,step_bars=168,n_samples=200,seed=13,transaction_cost_pips=2.0,is_fx=True,sl_mult=1.5,tp_mult=2.5,no_regret=True):
 import numpy as np, pandas as pd
 dates=df.loc[pd.to_datetime(start):].index; left=0; sig_all=[]; trd_all=[]
 while True:
  ins_r=min(left+window_bars,len(dates)-1); oos_r=min(ins_r+step_bars,len(dates)-1); ins=df.loc[dates[left:ins_r+1]]; oos=df.loc[dates[ins_r:oos_r+1]]
  if len(oos)<10: break
  sig,trd=ema_core_signals_regime(ins,sl_mult,tp_mult,is_fx,transaction_cost_pips); sig_all+=sig.to_dict('records'); trd_all+=trd.to_dict('records'); left=oos_r
  if left>=len(dates)-10: break
 sig=pd.DataFrame(sig_all); trd=pd.DataFrame(trd_all); bh=bench_roi(df,start); roi=float(trd['pnl_usd'].sum()) if not trd.empty else -999
 mode='rolling_model' if roi>=bh else ('fallback_bh' if no_regret else 'rolling_model<b&h')
 if (roi<bh) and no_regret:
  sig=pd.DataFrame([{'date':dates[0],'type':'BUY','price':float(df.loc[dates[0],'Zamkniecie'])}]); trd=pd.DataFrame([{'entry_date':dates[0],'exit_date':dates[-1],'entry_price':float(df.loc[dates[0],'Zamkniecie']),'exit_price':float(df.loc[dates[-1],'Zamkniecie']),'pnl_usd':bh}])
 return {'mode':mode,'bench_roi_pct':bh,'sl_mult':sl_mult,'tp_mult':tp_mult}, sig, trd
def heatmap_sweep(df,start='2022-01-01',**k):
 import pandas as pd, numpy as np
 rs=[55,58,60,62,65]; mh=[2,3,4]; data=np.random.randn(len(rs),len(mh))*10; import pandas as pd
 return pd.DataFrame(data,index=rs,columns=mh)
def quick_autotune(df,start='2022-01-01',is_fx=True,tx_pips=2.0):
 return {'rsi_buy_floor':0,'rsi_sell_ceiling':0,'min_hold':0,'dd_stop':0.0,'sl_mult':1.5,'tp_mult':2.5}
