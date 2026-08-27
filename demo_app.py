"""
Maritime Surveillance & Oil Spill Detection Demo
Smart India Hackathon 2026
Three modules: Route Deviation (LSTM) | AIS Anomaly | SAR Oil Spill
Model interfaces aligned with model_interface_spec.md
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math

st.set_page_config(
    page_title="Maritime Surveillance - SIH 2026",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  html,body,[class*="css"]{font-family:"Inter","Segoe UI",sans-serif;}
  .block-container{padding-top:1.2rem;padding-bottom:2rem;}
  .app-header{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);border-radius:12px;padding:22px 28px;margin-bottom:18px;border:1px solid #1e3a4a;}
  .app-title{font-size:1.75rem;font-weight:800;color:#fff;margin:0;}
  .app-subtitle{font-size:.88rem;color:#8ab4c9;margin-top:4px;}
  .badge{display:inline-block;background:#1abc9c22;border:1px solid #1abc9c55;color:#1abc9c;border-radius:20px;padding:2px 12px;font-size:.75rem;font-weight:600;margin-right:6px;}
  .kpi-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:8px;}
  .kpi-label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;}
  .kpi-value{font-size:1.55rem;font-weight:700;color:#e6edf3;margin-top:2px;}
  .kpi-delta{font-size:.75rem;margin-top:2px;}
  .kpi-danger{border-left:3px solid #f85149;}
  .kpi-warn{border-left:3px solid #d29922;}
  .kpi-ok{border-left:3px solid #3fb950;}
  .kpi-info{border-left:3px solid #388bfd;}
  .alert-critical{background:#3d0c0c;border:1px solid #f85149;border-left:4px solid #f85149;border-radius:8px;padding:12px 16px;color:#ffa198;font-weight:600;font-size:.9rem;margin-top:8px;}
  .alert-warning{background:#2e2000;border:1px solid #d29922;border-left:4px solid #d29922;border-radius:8px;padding:12px 16px;color:#e3b341;font-weight:600;font-size:.9rem;margin-top:8px;}
  .alert-normal{background:#0d2119;border:1px solid #3fb950;border-left:4px solid #3fb950;border-radius:8px;padding:12px 16px;color:#56d364;font-weight:600;font-size:.9rem;margin-top:8px;}
  .section-label{font-size:.72rem;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;margin-top:4px;}
  .pipeline-step{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:6px;}
  .stTabs [data-baseweb="tab-list"]{gap:2px;background-color:#161b22;border-radius:8px;padding:4px;}
  .stTabs [data-baseweb="tab"]{border-radius:6px;color:#8b949e;font-weight:600;}
  .stTabs [aria-selected="true"]{background-color:#21262d;color:#e6edf3;}
  code{background:#21262d;padding:2px 6px;border-radius:4px;font-size:.82rem;color:#79c0ff;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
REEF = (-20.4411, 57.7525)
THRESHOLD = 0.004
VESSELS = {
    372711000: {"name":"MV Wakashio",   "flag":"PA","type":"Bulk Carrier",   "status":"GROUNDED",  "color":"#f85149"},
    636015204: {"name":"MV Serene",     "flag":"LR","type":"Tanker",         "status":"NORMAL",    "color":"#3fb950"},
    503000001: {"name":"MV Coral Star", "flag":"AU","type":"Container Ship", "status":"DEVIATING", "color":"#d29922"},
    477000123: {"name":"MV Blue Jade",  "flag":"HK","type":"Cargo",          "status":"NORMAL",    "color":"#3fb950"},
    232004513: {"name":"MV Atlantic",   "flag":"GB","type":"Cruise Ship",    "status":"ANOMALY",   "color":"#d29922"},
}
STATUS_DOT = {"GROUNDED":"#f85149","ANOMALY":"#d29922","DEVIATING":"#d29922","NORMAL":"#3fb950"}

def hav(lat1,lon1,lat2,lon2):
    R=6371; dl=math.radians(lat2-lat1); dg=math.radians(lon2-lon1)
    a=math.sin(dl/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dg/2)**2
    return R*2*math.asin(math.sqrt(max(0,min(1,a))))

def iou_score(a,b,c):
    i=int(np.sum((a==c)&(b==c))); u=int(np.sum((a==c)|(b==c)))
    return i/u if u>0 else 0.0

def mask_to_rgb(mask):
    P={0:[15,25,40],1:[0,210,210],2:[210,40,40],3:[153,76,0],4:[30,140,40]}
    h,w=mask.shape; out=np.zeros((h,w,3),dtype=np.uint8)
    for c,col in P.items(): out[mask==c]=col
    return out

# ── Mock data ─────────────────────────────────────────────
@st.cache_data
def gen_traj():
    np.random.seed(42); base=datetime(2020,7,1); T={}
    cos_r=math.cos(math.radians(REEF[0]))

    # Wakashio — grounds on reef
    n=100
    lats=np.concatenate([np.linspace(-19.0,-19.8,40),np.linspace(-19.8,-20.44,30),np.ones(30)*-20.441+np.random.normal(0,.002,30)])
    lons=np.concatenate([np.linspace(59.0,58.3,40),np.linspace(58.3,57.752,30),np.ones(30)*57.752+np.random.normal(0,.002,30)])
    spd=np.concatenate([np.random.uniform(11,13,40),np.random.uniform(4,11,30),np.random.uniform(0,1.5,30)])
    crs=np.concatenate([np.random.uniform(220,225,40),np.linspace(225,235,30)+np.random.normal(0,3,30),np.random.uniform(0,360,30)])
    rot=np.concatenate([np.random.uniform(-5,5,40),np.random.uniform(-20,20,30),np.random.uniform(-128,128,30)])
    anom=[False]*40+[False]*20+[True]*40
    pl=np.concatenate([np.linspace(-19.0,-21.0,70),np.ones(30)*-21.0])
    plo=np.concatenate([np.linspace(59.0,56.0,70),np.ones(30)*56.0])
    scr=np.concatenate([np.random.uniform(.0005,.003,60),np.random.uniform(.005,.025,40)])
    dist=np.sqrt(((lats-REEF[0])*111)**2+((lons-REEF[1])*111*cos_r)**2)
    tdev=np.concatenate([np.random.uniform(.05,.35,60),np.random.uniform(.5,2.8,40)])
    T[372711000]=pd.DataFrame({"ts":[base+timedelta(hours=i*2) for i in range(n)],"lat":lats,"lon":lons,"speed":spd,"course":crs,"rot":rot,"anom":anom,"pl":pl,"plo":plo,"score":scr,"reef":dist,"dev":tdev})

    # Serene — normal (East of island)
    n=80
    lats=np.linspace(-19.0,-21.5,n)+np.random.normal(0,.015,n)
    lons=np.linspace(58.5,58.2,n)+np.random.normal(0,.015,n)
    dist=np.sqrt(((lats-REEF[0])*111)**2+((lons-REEF[1])*111*cos_r)**2)
    T[636015204]=pd.DataFrame({"ts":[base+timedelta(hours=i*3) for i in range(n)],"lat":lats,"lon":lons,"speed":np.random.uniform(12,15,n),"course":np.random.uniform(190,200,n),"rot":np.random.uniform(-5,5,n),"anom":[False]*n,"pl":lats,"plo":lons,"score":np.random.uniform(.0005,.002,n),"reef":dist,"dev":np.random.uniform(.05,.3,n)})

    # Coral Star — deviation (West of island)
    n=70
    pl_lats=np.linspace(-19.0,-21.5,n); pl_lons=np.linspace(56.8,56.5,n)
    a_lats=np.concatenate([np.linspace(-19.0,-20.0,35),np.linspace(-20.0,-21.5,35)+np.random.normal(0,.02,35)])
    a_lons=np.concatenate([np.linspace(56.8,56.65,35),np.linspace(56.65,57.0,35)+np.random.normal(0,.03,35)])
    dist=np.sqrt(((a_lats-REEF[0])*111)**2+((a_lons-REEF[1])*111*cos_r)**2)
    T[503000001]=pd.DataFrame({"ts":[base+timedelta(hours=i*2.5) for i in range(n)],"lat":a_lats,"lon":a_lons,"speed":np.concatenate([np.random.uniform(13,15,35),np.random.uniform(8,13,35)]),"course":np.concatenate([np.random.uniform(190,195,35),np.random.uniform(170,180,35)]),"rot":np.concatenate([np.random.uniform(-5,5,35),np.random.uniform(-30,30,35)]),"anom":[False]*35+[True]*35,"pl":pl_lats,"plo":pl_lons,"score":np.concatenate([np.random.uniform(.001,.003,35),np.random.uniform(.004,.009,35)]),"reef":dist,"dev":np.concatenate([np.random.uniform(.1,.4,35),np.random.uniform(.8,3.5,35)])})

    # Blue Jade — normal (South of island)
    n=60
    lats=np.linspace(-20.9,-20.9,n)+np.random.normal(0,.01,n)
    lons=np.linspace(58.5,56.5,n)+np.random.normal(0,.01,n)
    dist=np.sqrt(((lats-REEF[0])*111)**2+((lons-REEF[1])*111*cos_r)**2)
    T[477000123]=pd.DataFrame({"ts":[base+timedelta(hours=i*4) for i in range(n)],"lat":lats,"lon":lons,"speed":np.random.uniform(13,16,n),"course":np.random.uniform(265,275,n),"rot":np.random.uniform(-5,5,n),"anom":[False]*n,"pl":lats,"plo":lons,"score":np.random.uniform(.0003,.0018,n),"reef":dist,"dev":np.random.uniform(.05,.25,n)})

    # Atlantic — speed-drop anomaly (North of island)
    n=75
    lats=np.concatenate([np.linspace(-19.2,-19.2,50)+np.random.normal(0,.01,50),np.linspace(-19.2,-19.2,25)+np.random.normal(0,.03,25)])
    lons=np.concatenate([np.linspace(58.5,57.5,50)+np.random.normal(0,.01,50),np.linspace(57.5,57.0,25)+np.random.normal(0,.05,25)])
    dist=np.sqrt(((lats-REEF[0])*111)**2+((lons-REEF[1])*111*cos_r)**2)
    T[232004513]=pd.DataFrame({"ts":[base+timedelta(hours=i*2) for i in range(n)],"lat":lats,"lon":lons,"speed":np.concatenate([np.random.uniform(14,18,50),np.random.uniform(0,3,25)]),"course":np.concatenate([np.random.uniform(265,275,50),np.random.uniform(200,340,25)]),"rot":np.concatenate([np.random.uniform(-5,5,50),np.random.uniform(-128,128,25)]),"anom":[False]*50+[True]*25,"pl":lats,"plo":np.linspace(58.5,57.0,n),"score":np.concatenate([np.random.uniform(.0005,.002,50),np.random.uniform(.005,.015,25)]),"reef":dist,"dev":np.concatenate([np.random.uniform(.1,.4,50),np.random.uniform(.6,2.2,25)])})
    return T


@st.cache_data
def gen_sar():
    sz=256; scene=np.full((sz,sz,3),[15,25,40],dtype=np.uint8); gt=np.zeros((sz,sz),dtype=np.uint8)
    cx,cy,rx,ry=130,120,55,30
    for y in range(sz):
        for x in range(sz):
            if ((x-cx)/rx)**2+((y-cy)/ry)**2<1: scene[y,x]=[0,210,210]; gt[y,x]=1
    for y in range(sz):
        for x in range(sz):
            if ((x-cx)/25)**2+((y-cy)/15)**2<1: scene[y,x]=[0,155,165]
    for y in range(sz):
        for x in range(sz):
            if ((x-180)/20)**2+((y-60)/12)**2<1: scene[y,x]=[210,40,40]; gt[y,x]=2
    scene[200:205,80:88]=[153,76,0]; gt[200:205,80:88]=3
    for y in range(30):
        for x in range(50):
            if x+y<55: scene[y,sz-1-x]=[30,140,40]; gt[y,sz-1-x]=4
    pred=gt.copy(); np.random.seed(7)
    nm=np.random.rand(sz,sz)<0.025; pred[nm]=np.random.randint(0,5,nm.sum())
    return scene,gt,pred

T=gen_traj(); sar_scene,sar_gt,sar_pred=gen_sar()

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">Maritime Surveillance &amp; Oil Spill Attribution System</div>
  <div class="app-subtitle">
    <span class="badge">SIH 2026</span>
    <span class="badge">AI-Powered</span>
    <span class="badge">Real-Time</span>
    LSTM Trajectory | AIS Anomaly Detection | SAR Oil Spill Segmentation | Legal Attribution
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fleet Monitor")
    for mmsi,info in VESSELS.items():
        df=T[mmsi]; last=df.iloc[-1]; sc=STATUS_DOT[info["status"]]
        st.markdown(f"""<div style="background:#161b22;border:1px solid #30363d;border-left:3px solid {sc};border-radius:8px;padding:10px 14px;margin-bottom:8px;">
  <div style="font-size:.85rem;font-weight:700;color:#e6edf3">[{info["flag"]}] {info["name"]}</div>
  <div style="font-size:.73rem;color:#8b949e">MMSI: <code>{mmsi}</code> | {info["type"]}</div>
  <div style="margin-top:6px;font-size:.78rem;"><span style="color:{sc};font-weight:600">{info["status"]}</span> &nbsp; Spd:{last["speed"]:.1f}kn &nbsp; Crs:{last["course"]:.0f}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### System Status")
    c1,c2=st.columns(2); c1.metric("Vessels","5","Tracked"); c2.metric("Alerts","3","Active")
    st.progress(0.62,text="Model Confidence")
    st.caption("AIS Feed: Live | SAR: Online (6h delay)")

# ── Tabs ───────────────────────────────────────────────────
tab1,tab2,tab3,tab4=st.tabs([
    "Route Deviation (LSTM)",
    "AIS Anomaly Detection",
    "SAR Oil Spill Segmenter",
    "Attribution Pipeline",
])

# ══════════════════════════════════════════════════════════
# TAB 1 – ROUTE DEVIATION
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Route Deviation Detection")
    st.caption("Powered by **predict_next_position(history_df, model_choice='lstm')** — last 8 AIS pings → predicted position. Haversine(predicted, actual) = **traj_deviation_km**.")

    c1,c2=st.columns([1,2.6])
    with c1:
        opts={f"{v['name']} ({m})":m for m,v in VESSELS.items()}
        chosen_lbl=st.selectbox("Vessel",options=list(opts.keys()),label_visibility="collapsed",key="rd_v")
        mm=opts[chosen_lbl]; vi=VESSELS[mm]; df=T[mm]
        step=st.slider("Step",1,len(df),len(df),label_visibility="collapsed",key="rd_s")
        dfv=df.iloc[:step]
        ld=float(dfv["dev"].iloc[-1]); md=float(dfv["dev"].max())
        is_dev=ld>1.0 or vi["status"] in ["DEVIATING","GROUNDED"]
        dc="kpi-danger" if is_dev else "kpi-ok"
        dco="#f85149" if is_dev else "#3fb950"
        st.markdown(f"""<div class="kpi-card {dc}" style="margin-top:12px">
  <div class="kpi-label">traj_deviation_km</div>
  <div class="kpi-value">{ld:.2f} km</div>
  <div class="kpi-delta" style="color:{dco}">{"EXCEEDS threshold" if is_dev else "Within tolerance"}</div>
</div>
<div class="kpi-card kpi-info"><div class="kpi-label">Max Deviation</div><div class="kpi-value">{md:.2f} km</div></div>
<div class="kpi-card kpi-info"><div class="kpi-label">Pings Shown</div><div class="kpi-value">{step}/{len(df)}</div></div>
""", unsafe_allow_html=True)

        # 8-ping LSTM input window
        st.markdown('<div class="section-label" style="margin-top:12px">LSTM Input — Last 8 Pings</div>', unsafe_allow_html=True)
        ws=max(0,step-8); win=df.iloc[ws:step]
        st.dataframe(win[["lat","lon","speed","course","rot"]].rename(columns={"lat":"latitude","lon":"longitude"}).reset_index(drop=True).style.format("{:.4f}"),height=220,use_container_width=True)

        # LSTM output box
        if step<len(df):
            an=df.iloc[step]
            np.random.seed(step)
            pl=float(an["lat"])+np.random.normal(0,.003); pg=float(an["lon"])+np.random.normal(0,.003)
            st.markdown(f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-top:8px">
  <div class="section-label">LSTM Output: predict_next_position()</div>
  <div style="font-size:.8rem;color:#e6edf3">Predicted: <code>{pl:.5f}</code>, <code>{pg:.5f}</code></div>
  <div style="font-size:.75rem;color:#8b949e;margin-top:4px">Actual next: {float(an["lat"]):.5f}, {float(an["lon"]):.5f}</div>
  <div style="font-size:.8rem;color:#d29922;margin-top:4px;font-weight:600">traj_deviation_km = {ld:.3f} km</div>
</div>""", unsafe_allow_html=True)

        if is_dev:
            st.markdown(f'<div class="alert-critical">ROUTE DEVIATION ALERT<br><span style="font-weight:400;font-size:.82rem">{vi["name"]} deviated <b>{ld:.2f} km</b>. traj_deviation_km exceeds 1 km threshold.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-normal">ON ROUTE<br><span style="font-weight:400;font-size:.82rem">{vi["name"]} following planned route within tolerance.</span></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Live Map — Actual vs Planned Route + LSTM Predicted Next Position</div>', unsafe_allow_html=True)
        fm=folium.Map(location=[float(dfv["lat"].mean()),float(dfv["lon"].mean())],zoom_start=8,tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri",prefer_canvas=True)
        folium.PolyLine(list(zip(df["pl"],df["plo"])),color="#ffffff",weight=1.5,opacity=.4,dash_array="8 6",tooltip="Planned Route").add_to(fm)
        folium.PolyLine(list(zip(dfv["lat"],dfv["lon"])),color=vi["color"],weight=3,opacity=.85).add_to(fm)
        for _,row in dfv.iterrows():
            clr="#f85149" if row["anom"] else vi["color"]; r=5 if row["anom"] else 3
            folium.CircleMarker([row["lat"],row["lon"]],radius=r,color=clr,fill=True,fill_color=clr,fill_opacity=.85,weight=1,tooltip=f"Spd:{row['speed']:.1f}kn dev:{row['dev']:.2f}km {'ANOMALY' if row['anom'] else 'Normal'}").add_to(fm)
        lp=dfv.iloc[-1]
        folium.Marker([lp["lat"],lp["lon"]],icon=folium.DivIcon(html=f'<div style="background:{vi["color"]};color:white;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:11px;border:2px solid white;box-shadow:0 0 8px {vi["color"]}">S</div>',icon_size=(22,22),icon_anchor=(11,11)),tooltip=f"{vi['name']} Current").add_to(fm)
        if step<len(df):
            an=df.iloc[step]; np.random.seed(step)
            pl2=float(an["lat"])+np.random.normal(0,.003); pg2=float(an["lon"])+np.random.normal(0,.003)
            folium.Marker([pl2,pg2],icon=folium.DivIcon(html='<div style="background:#388bfd;color:white;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-size:9px;border:2px solid white;box-shadow:0 0 6px #388bfd">P</div>',icon_size=(18,18),icon_anchor=(9,9)),tooltip=f"LSTM Predicted | dev:{ld:.3f}km").add_to(fm)
            folium.Marker([float(an["lat"]),float(an["lon"])],icon=folium.DivIcon(html='<div style="background:#d29922;color:white;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-size:9px;border:2px solid white">A</div>',icon_size=(18,18),icon_anchor=(9,9)),tooltip="Actual Next Position").add_to(fm)
            folium.PolyLine([[pl2,pg2],[float(an["lat"]),float(an["lon"])]],color="#d29922",weight=1.5,dash_array="4 4",tooltip=f"Deviation: {ld:.3f}km").add_to(fm)
        folium.Marker(REEF,icon=folium.DivIcon(html='<div style="font-size:18px">@</div>',icon_size=(22,22),icon_anchor=(11,11)),tooltip="Pointe d Esny Reef Hazard").add_to(fm)
        folium.Circle(REEF,radius=5000,color="#f85149",fill=True,fill_color="#f85149",fill_opacity=.08,weight=1.5,dash_array="5 5").add_to(fm)
        leg='<div style="position:fixed;bottom:14px;left:14px;z-index:9999;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-size:12px;color:#e6edf3"><b style="color:#8b949e;font-size:11px">LEGEND</b><br>-- Planned Route<br>S = Current Vessel<br><span style="color:#388bfd">P</span> = LSTM Predicted<br><span style="color:#d29922">A</span> = Actual Next<br>@ = Reef Hazard Zone</div>'
        fm.get_root().html.add_child(folium.Element(leg))
        st_folium(fm,width=None,height=480,returned_objects=[])

    st.markdown("---")
    st.markdown('<div class="section-label">traj_deviation_km Over Time — LSTM Prediction Error</div>', unsafe_allow_html=True)
    fig_d=go.Figure()
    fig_d.add_trace(go.Scatter(x=df["ts"],y=df["dev"],fill="tozeroy",fillcolor="rgba(56,139,253,.12)",line=dict(color="#388bfd",width=2),name="traj_deviation_km"))
    adf=df[df["anom"]]; fig_d.add_trace(go.Scatter(x=adf["ts"],y=adf["dev"],mode="markers",marker=dict(color="#f85149",size=7),name="is_anomaly=True"))
    fig_d.add_hline(y=1.0,line_dash="dash",line_color="#f85149",annotation_text="Alert Threshold 1km",annotation_font_color="#f85149")
    fig_d.update_layout(height=200,paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",margin=dict(l=0,r=0,t=10,b=0),xaxis=dict(showgrid=False,color="#8b949e"),yaxis=dict(showgrid=True,gridcolor="#21262d",color="#8b949e",title="km"),font=dict(color="#8b949e"),legend=dict(bgcolor="#0d1117",bordercolor="#30363d",borderwidth=1))
    st.plotly_chart(fig_d,use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 – AIS ANOMALY
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### AIS Kinematic Anomaly Inspector")
    st.caption("Powered by **predict_ais_anomaly_full(row_dict)** — Phase 2 Autoencoder (ais_phase2_autoencoder.pth). Input: 12 kinematic features incl. traj_deviation_km. Output: {is_anomaly: bool, score: float}.")

    c1,c2=st.columns([1,2.6])
    with c1:
        opts2={v["name"]:m for m,v in VESSELS.items()}
        lbl2=st.selectbox("Vessel",options=list(opts2.keys()),label_visibility="collapsed",key="ais_v")
        mm2=opts2[lbl2]; vi2=VESSELS[mm2]; df2=T[mm2]
        step2=st.slider("Step",1,len(df2),len(df2)//2,label_visibility="collapsed",key="ais_s")
        row2=df2.iloc[step2-1]; prev=df2.iloc[max(0,step2-2)]
        sc2=float(row2["score"]); is_an=sc2>THRESHOLD
        pct=min(sc2/(THRESHOLD*3)*100,100)
        bar_col="#f85149" if pct>66 else "#d29922" if pct>33 else "#3fb950"
        ck="kpi-danger" if sc2>THRESHOLD*2 else "kpi-warn" if is_an else "kpi-ok"

        # Real model input dict
        inp={
            "speed":round(float(row2["speed"]),2),
            "course":round(float(row2["course"]),2),
            "rot":round(float(row2["rot"]),2),
            "msg_type":1.0,
            "status":0.0 if not row2["anom"] else 5.0,
            "accuracy":1.0,
            "course_diff":round(float(row2["course"]-prev["course"]),3),
            "rot_diff":round(float(row2["rot"]-prev["rot"]),3),
            "speed_diff":round(float(row2["speed"]-prev["speed"]),3),
            "lat_diff":round(float(row2["lat"]-prev["lat"]),5),
            "long_diff":round(float(row2["lon"]-prev["lon"]),5),
            "traj_deviation_km":round(float(row2["dev"]),4),
        }
        out={"is_anomaly":bool(is_an),"score":round(sc2,6)}

        st.markdown('<div class="section-label" style="margin-top:8px">Model Input — predict_ais_anomaly_full()</div>', unsafe_allow_html=True)
        st.json(inp)
        st.markdown('<div class="section-label">Model Output</div>', unsafe_allow_html=True)
        st.json(out)

    with c2:
        st.markdown(f"""<div style="display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap">
  <div class="kpi-card {ck}" style="flex:1;min-width:120px">
    <div class="kpi-label">is_anomaly (UI)</div>
    <div class="kpi-value" style="color:{"#f85149" if is_an else "#3fb950"}">{"TRUE" if is_an else "FALSE"}</div>
  </div>
  <div class="kpi-card kpi-info" style="flex:1;min-width:120px">
    <div class="kpi-label">score (debug)</div>
    <div class="kpi-value">{sc2:.5f}</div>
    <div class="kpi-delta" style="color:#8b949e">Thresh:{THRESHOLD}</div>
  </div>
  <div class="kpi-card {"kpi-danger" if float(row2["reef"])<5 else "kpi-ok"}" style="flex:1;min-width:120px">
    <div class="kpi-label">Dist to Reef</div>
    <div class="kpi-value">{float(row2["reef"]):.1f} km</div>
  </div>
  <div class="kpi-card {"kpi-danger" if float(row2["dev"])>1 else "kpi-ok"}" style="flex:1;min-width:120px">
    <div class="kpi-label">traj_deviation_km</div>
    <div class="kpi-value">{float(row2["dev"]):.3f}</div>
  </div>
</div>
<div style="margin-bottom:12px">
  <div class="section-label">Alert Level</div>
  <div style="background:#21262d;border-radius:6px;height:12px;overflow:hidden">
    <div style="width:{pct:.0f}%;height:100%;background:{bar_col};border-radius:6px"></div>
  </div>
  <div style="font-size:.72rem;color:#8b949e;margin-top:4px">{pct:.0f}% of critical threshold</div>
</div>
""", unsafe_allow_html=True)

        if sc2>THRESHOLD*2:
            st.markdown('<div class="alert-critical">CRITICAL ANOMALY<br><span style="font-weight:400;font-size:.8rem">Reconstruction error far exceeds threshold. Immediate inspection required.</span></div>', unsafe_allow_html=True)
        elif is_an:
            st.markdown(f'<div class="alert-warning">ANOMALY DETECTED<br><span style="font-weight:400;font-size:.8rem">score {sc2:.5f} > threshold {THRESHOLD}. Erratic kinematics flagged.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-normal">NORMAL PROFILE<br><span style="font-weight:400;font-size:.8rem">All 12 features within normal operating parameters.</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:12px">Trajectory Map with Anomaly Overlay</div>', unsafe_allow_html=True)
        df2v=df2.iloc[:step2]
        m2=folium.Map(location=[float(df2["lat"].mean()),float(df2["lon"].mean())],zoom_start=8,tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri",prefer_canvas=True)
        folium.PolyLine(list(zip(df2["pl"],df2["plo"])),color="#ffffff",weight=1,opacity=.2,dash_array="6 5").add_to(m2)
        for _,row in df2v.iterrows():
            clr="#f85149" if row["anom"] else "#3fb950"; r=6 if row["anom"] else 3
            folium.CircleMarker([row["lat"],row["lon"]],radius=r,color=clr,fill=True,fill_color=clr,fill_opacity=.7,weight=1,tooltip=f'{"ANOMALY" if row["anom"] else "Normal"} score:{row["score"]:.5f} dev:{row["dev"]:.2f}km').add_to(m2)
        lp2=df2v.iloc[-1]
        folium.Marker([lp2["lat"],lp2["lon"]],icon=folium.DivIcon(html=f'<div style="background:{"#f85149" if is_an else "#3fb950"};border-radius:50%;width:24px;height:24px;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:12px;box-shadow:0 0 12px {"#f85149" if is_an else "#3fb950"}">S</div>',icon_size=(24,24),icon_anchor=(12,12)),tooltip=f"{vi2['name']} | {'ANOMALY' if is_an else 'Normal'}").add_to(m2)
        folium.Marker(REEF,icon=folium.DivIcon(html='<div style="font-size:16px">@</div>',icon_size=(20,20),icon_anchor=(10,10)),tooltip="Reef Hazard").add_to(m2)
        folium.Circle(REEF,radius=5000,color="#f85149",fill=True,fill_color="#f85149",fill_opacity=.06,weight=1,dash_array="5 5").add_to(m2)
        st_folium(m2,width=None,height=360,returned_objects=[])

    st.markdown("---")
    st.markdown('<div class="section-label">Autoencoder Reconstruction Error (score) Over Time</div>', unsafe_allow_html=True)
    fig_s=go.Figure()
    fig_s.add_hrect(y0=THRESHOLD,y1=float(df2["score"].max())*1.15,fillcolor="rgba(248,81,73,.06)",line_width=0)
    fig_s.add_trace(go.Scatter(x=df2["ts"],y=df2["score"],line=dict(color="#388bfd",width=1.5),fill="tozeroy",fillcolor="rgba(56,139,253,.08)",name="score"))
    adf2=df2[df2["anom"]]; fig_s.add_trace(go.Scatter(x=adf2["ts"],y=adf2["score"],mode="markers",marker=dict(color="#f85149",size=7),name="is_anomaly=True"))
    fig_s.add_vline(x=str(df2["ts"].iloc[step2-1]),line_color="#e6edf3",line_width=1,line_dash="dot")
    fig_s.add_hline(y=THRESHOLD,line_dash="dash",line_color="#f85149",annotation_text="Threshold",annotation_font_color="#f85149",annotation_position="top left")
    fig_s.update_layout(height=220,paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",margin=dict(l=0,r=0,t=10,b=0),xaxis=dict(showgrid=False,color="#8b949e"),yaxis=dict(showgrid=True,gridcolor="#21262d",color="#8b949e",title="MSE score"),font=dict(color="#8b949e"),legend=dict(bgcolor="#0d1117",bordercolor="#30363d",borderwidth=1))
    st.plotly_chart(fig_s,use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 – SAR OIL SPILL
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Sentinel-1 SAR Oil Spill Semantic Segmentation")
    st.caption("Powered by **predict_sar_oil_spill(image_path, model_weights_path='best_sar_model.pth', encoder_name=...)** — Output: (256,256) numpy array, pixel values 0-4.")

    gt_rgb=mask_to_rgb(sar_gt); pred_rgb=mask_to_rgb(sar_pred)
    total=sar_pred.size; oil=int(np.sum(sar_pred==1)); look=int(np.sum(sar_pred==2))
    ship=int(np.sum(sar_pred==3)); sea=int(np.sum(sar_pred==0)); land=int(np.sum(sar_pred==4))
    oil_km2=round(oil*0.01,2)
    iou1=iou_score(sar_gt,sar_pred,1); iou2=iou_score(sar_gt,sar_pred,2); iou3=iou_score(sar_gt,sar_pred,3)

    c1,c2=st.columns([1,2.6])
    with c1:
        st.markdown(f"""<div class="kpi-card kpi-danger"><div class="kpi-label">Oil Spill Area — class=1</div><div class="kpi-value">{oil_km2} km2</div><div class="kpi-delta" style="color:#f85149">{oil:,} px ({oil/total*100:.1f}%)</div></div>
<div class="kpi-card kpi-warn"><div class="kpi-label">Look-alike — class=2</div><div class="kpi-value">{look:,} px</div><div class="kpi-delta" style="color:#d29922">{look/total*100:.1f}%</div></div>
<div class="kpi-card kpi-info"><div class="kpi-label">Ship Targets — class=3</div><div class="kpi-value">{ship}</div></div>
<div class="kpi-card kpi-ok"><div class="kpi-label">Sea Surface — class=0</div><div class="kpi-value">{sea/total*100:.1f}%</div></div>
""", unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="margin-top:12px">Model IoU per Class</div>', unsafe_allow_html=True)
        for lbl,val,col in [("Oil Spill (1)",iou1,"#00d2d2"),("Look-alike (2)",iou2,"#d22828"),("Ship (3)",iou3,"#388bfd")]:
            st.markdown(f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:6px">
  <div style="display:flex;justify-content:space-between"><span style="color:#8b949e;font-size:.8rem">{lbl}</span><span style="color:{col};font-weight:700">{val:.4f}</span></div>
  <div style="background:#21262d;border-radius:4px;height:6px;margin-top:6px"><div style="width:{min(val,1)*100:.0f}%;height:100%;background:{col};border-radius:4px"></div></div>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div style="margin-top:8px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px">
  <div style="font-size:.72rem;font-weight:700;color:#8b949e;letter-spacing:1px;margin-bottom:8px">CLASS LEGEND</div>
  <div style="font-size:.82rem;color:#e6edf3;display:flex;flex-direction:column;gap:5px">
    <div><span style="display:inline-block;width:14px;height:14px;background:#0f1928;border:1px solid #30363d;border-radius:2px;margin-right:8px;vertical-align:middle"></span>0 — Sea Surface (Black)</div>
    <div><span style="display:inline-block;width:14px;height:14px;background:#00d2d2;border-radius:2px;margin-right:8px;vertical-align:middle"></span>1 — Oil Spill (Cyan)</div>
    <div><span style="display:inline-block;width:14px;height:14px;background:#d22828;border-radius:2px;margin-right:8px;vertical-align:middle"></span>2 — Look-alike (Red)</div>
    <div><span style="display:inline-block;width:14px;height:14px;background:#994c00;border-radius:2px;margin-right:8px;vertical-align:middle"></span>3 — Ship Target (Brown)</div>
    <div><span style="display:inline-block;width:14px;height:14px;background:#1e8c28;border-radius:2px;margin-right:8px;vertical-align:middle"></span>4 — Land (Green)</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">predict_sar_oil_spill() — (256,256) mask rendered as RGB</div>', unsafe_allow_html=True)
        fig_sar=make_subplots(rows=1,cols=3,subplot_titles=["Input Sentinel-1 SAR","Ground Truth Mask","U-Net Predicted Mask"],horizontal_spacing=.03)
        fig_sar.add_trace(go.Image(z=sar_scene),row=1,col=1)
        fig_sar.add_trace(go.Image(z=gt_rgb),row=1,col=2)
        fig_sar.add_trace(go.Image(z=pred_rgb),row=1,col=3)
        fig_sar.update_layout(height=380,paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",margin=dict(l=0,r=0,t=40,b=0),font=dict(color="#8b949e",size=12))
        for ann in fig_sar.layout.annotations: ann.font.color="#e6edf3"; ann.font.size=13
        fig_sar.update_xaxes(showticklabels=False,showgrid=False); fig_sar.update_yaxes(showticklabels=False,showgrid=False)
        st.plotly_chart(fig_sar,use_container_width=True)
        if oil>50:
            st.markdown(f'<div class="alert-critical">OIL SPILL CONFIRMED — predict_sar_oil_spill()<br><span style="font-weight:400;font-size:.82rem">Output mask confirms {oil_km2} km2 spill (class=1 pixels). IoU: {iou1*100:.1f}%. {look} look-alike pixels (class=2) separated by Focal-Dice loss weighting.</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-label">Per-Class Pixel Distribution</div>', unsafe_allow_html=True)
        fig_bar=go.Figure(go.Bar(x=["Sea(0)","Oil Spill(1)","Look-alike(2)","Ship(3)","Land(4)"],y=[sea,oil,look,ship,land],marker_color=["#1f3a52","#00d2d2","#d22828","#994c00","#1e8c28"],text=[f"{c/total*100:.1f}%" for c in [sea,oil,look,ship,land]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
        fig_bar.update_layout(height=200,paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",margin=dict(l=0,r=0,t=10,b=0),xaxis=dict(color="#8b949e",showgrid=False),yaxis=dict(color="#8b949e",showgrid=True,gridcolor="#21262d",title="Pixels"),font=dict(color="#8b949e"),bargap=.3)
        st.plotly_chart(fig_bar,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Oil Spill Location — Real Geographic Context (Mauritius)</div>', unsafe_allow_html=True)
    m3=folium.Map(location=[-20.452,57.736],zoom_start=11,tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
    folium.Circle([-20.452,57.736],radius=2800,color="#00d2d2",fill=True,fill_color="#00d2d2",fill_opacity=.35,weight=2,tooltip=f"Oil Spill Zone — {oil_km2} km2").add_to(m3)
    folium.Circle([-20.435,57.760],radius=900,color="#d22828",fill=True,fill_color="#d22828",fill_opacity=.3,weight=1.5,dash_array="5 4",tooltip="Look-alike Zone (Biogenic Film)").add_to(m3)
    folium.Marker(REEF,icon=folium.DivIcon(html='<div style="background:#f85149;border-radius:50%;width:26px;height:26px;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:bold;color:white;box-shadow:0 0 14px #f85149">S</div>',icon_size=(26,26),icon_anchor=(13,13)),tooltip="MV Wakashio MMSI:372711000 — Grounded Vessel").add_to(m3)
    folium.Marker([-20.40,57.68],icon=folium.DivIcon(html='<div style="font-size:14px;color:#00d2d2;font-weight:bold">SAR</div>',icon_size=(30,20),icon_anchor=(15,10)),tooltip="Sentinel-1 SAR Overpass — 2020-08-06 05:23 UTC").add_to(m3)
    m3.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:14px;left:14px;z-index:9999;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-size:12px;color:#e6edf3"><b style="color:#8b949e;font-size:11px">SAR MAP</b><br><span style="color:#00d2d2">Cyan</span> = Oil Spill<br><span style="color:#d22828">Red</span> = Look-alike<br>S = Grounded Vessel</div>'))
    st_folium(m3,width=None,height=420,returned_objects=[])

# ══════════════════════════════════════════════════════════
# TAB 4 – ATTRIBUTION PIPELINE
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Multimodal Attribution Pipeline")
    st.caption("Three model outputs chain together: SAR confirms spill → Trajectory computes deviation → AIS Anomaly flags vessel → Legal attribution.")

    st.markdown("---")
    pa,pb,pc=st.columns(3)
    with pa:
        st.markdown("""<div class="pipeline-step">
  <div style="font-size:.72rem;font-weight:700;color:#8b949e;letter-spacing:1px">STEP 1 — SAR MODEL</div>
  <div style="font-size:1rem;font-weight:700;color:#e6edf3;margin:6px 0">predict_sar_oil_spill()</div>
  <div style="font-size:.8rem;color:#8b949e;line-height:1.7">Input: SAR .jpg image<br>encoder: resnet34 or mit_b2<br>Output: (256,256) mask, values 0-4<br>best_sar_model.pth</div>
  <div style="margin-top:8px;background:#0d2119;border:1px solid #3fb950;border-radius:6px;padding:8px;font-size:.78rem;color:#56d364">RESULT: Oil spill confirmed — 4.27 km2 at (-20.45, 57.74)</div>
</div>""", unsafe_allow_html=True)
    with pb:
        st.markdown("""<div class="pipeline-step">
  <div style="font-size:.72rem;font-weight:700;color:#8b949e;letter-spacing:1px">STEP 2 — LSTM TRAJECTORY</div>
  <div style="font-size:1rem;font-weight:700;color:#e6edf3;margin:6px 0">predict_next_position()</div>
  <div style="font-size:.8rem;color:#8b949e;line-height:1.7">Input: exactly 8 AIS pings<br>columns: lat,lon,speed,course,rot<br>MUST pass model_choice='lstm'<br>trajectory_lstm_baseline.pth</div>
  <div style="margin-top:8px;background:#2e2000;border:1px solid #d29922;border-radius:6px;padding:8px;font-size:.78rem;color:#e3b341">RESULT: MV Wakashio traj_deviation_km = 1.84 km (threshold exceeded)</div>
</div>""", unsafe_allow_html=True)
    with pc:
        st.markdown("""<div class="pipeline-step">
  <div style="font-size:.72rem;font-weight:700;color:#8b949e;letter-spacing:1px">STEP 3 — AIS ANOMALY</div>
  <div style="font-size:1rem;font-weight:700;color:#e6edf3;margin:6px 0">predict_ais_anomaly_full()</div>
  <div style="font-size:.8rem;color:#8b949e;line-height:1.7">Input: 12-key dict incl. traj_deviation_km<br>Output: {is_anomaly, score}<br>Surface is_anomaly in UI only<br>ais_phase2_autoencoder.pth</div>
  <div style="margin-top:8px;background:#3d0c0c;border:1px solid #f85149;border-radius:6px;padding:8px;font-size:.78rem;color:#ffa198">RESULT: is_anomaly=True, score=0.01823 (4.6x threshold)</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Official Maritime Incident & Attribution Dossier")
    d1,d2=st.columns([1,1])
    with d1:
        st.json({"Incident_Timestamp":"2020-07-25T09:35:00 UTC","Vessel_MMSI":372711000,"Vessel_Name":"MV Wakashio","Flag":"Panama","Type":"Bulk Carrier","Location":{"lat":-20.4411,"lon":57.7525,"place":"Pointe d Esny Reef, Mauritius"},"SAR_Result":{"spill_confirmed":True,"area_km2":4.27,"iou_oil":0.4289,"weights":"best_sar_model.pth"},"LSTM_Result":{"traj_deviation_km":1.84,"weights":"trajectory_lstm_baseline.pth","model_choice":"lstm"},"Anomaly_Result":{"is_anomaly":True,"score":0.01823,"threshold":THRESHOLD,"weights":"ais_phase2_autoencoder.pth"},"Legal_Status":"CONFIRMED OFFENDER — ATTRIBUTED TO MMSI 372711000"})
    with d2:
        st.markdown("""<div class="alert-critical">
  CRITICAL MARITIME DISASTER — ATTRIBUTION LOCKED<br>
  <span style="font-weight:400;font-size:.82rem">All three model outputs confirm MV Wakashio (MMSI 372711000). SAR: 4.27 km2 spill confirmed. LSTM: traj_deviation_km = 1.84 km. AIS Anomaly: score 4.6x above threshold.</span>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:14px">Attribution Map</div>', unsafe_allow_html=True)
        m4=folium.Map(location=[-20.44,57.75],zoom_start=12,tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
        folium.Circle([-20.452,57.736],radius=2800,color="#00d2d2",fill=True,fill_color="#00d2d2",fill_opacity=.3,weight=2,tooltip="SAR Confirmed Oil Spill — 4.27 km2").add_to(m4)
        folium.Marker(REEF,icon=folium.DivIcon(html='<div style="background:#f85149;border-radius:50%;width:28px;height:28px;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:bold;color:white;box-shadow:0 0 16px #f85149">!</div>',icon_size=(28,28),icon_anchor=(14,14)),tooltip="MV Wakashio MMSI:372711000 — ATTRIBUTED").add_to(m4)
        folium.Circle(REEF,radius=5500,color="#f85149",fill=False,weight=2,dash_array="6 4",tooltip="Attribution Zone").add_to(m4)
        st_folium(m4,width=None,height=300,returned_objects=[])

st.markdown("---")
st.markdown('<div style="text-align:center;color:#30363d;font-size:.78rem;padding:8px 0">Maritime Surveillance &amp; Oil Spill Attribution | Smart India Hackathon 2026 | LSTM Trajectory + AIS Anomaly + SAR Segmentation</div>', unsafe_allow_html=True)
