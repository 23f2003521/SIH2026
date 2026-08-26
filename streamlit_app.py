import os
import glob
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Maritime Surveillance & Oil Spill Attribution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
    }
    .alert-critical {
        background-color: #FEE2E2;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        font-weight: 600;
    }
    .alert-normal {
        background-color: #ECFDF5;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #10B981;
        color: #065F46;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
# PyTorch Model Definitions
# -------------------------------------------------------------
class TrajectoryAutoencoder(nn.Module):
    def __init__(self, input_dim=12):
        super(TrajectoryAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SAROilSpillUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super(SAROilSpillUNet, self).__init__()
        self.e1 = ConvBlock(in_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        c4 = self.e4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))
        d4 = self.d4(torch.cat([self.up4(bn), c4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), c1], dim=1))
        return self.out_conv(d1)

CLASS_NAMES = ["Sea Surface", "Oil Spill", "Look-alike", "Ship", "Land"]
PALETTE = {
    0: (0, 0, 0),       # Black
    1: (0, 255, 255),   # Cyan
    2: (255, 0, 0),     # Red
    3: (153, 76, 0),    # Brown
    4: (0, 153, 0)      # Green
}

def mask_to_rgb(mask_1d):
    h, w = mask_1d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask_1d == class_id] = color
    return rgb

# -------------------------------------------------------------
# Cached Loaders
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    scaler_path = "models/ais_scaler.joblib"
    thresh_path = "models/ais_threshold.json"
    ae_weights = "models/ais_autoencoder.pth"
    unet_weights = "models/sar_unet.pth"
    
    ais_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    opt_thresh = 0.004688
    if os.path.exists(thresh_path):
        try:
            with open(thresh_path) as f:
                opt_thresh = json.load(f).get("optimal_threshold", 0.004688)
        except Exception:
            pass
            
    ae_model = TrajectoryAutoencoder(input_dim=12).to(device)
    if os.path.exists(ae_weights):
        ae_model.load_state_dict(torch.load(ae_weights, map_location=device))
    ae_model.eval()
    
    sar_model = SAROilSpillUNet(num_classes=5).to(device)
    if os.path.exists(unet_weights):
        sar_model.load_state_dict(torch.load(unet_weights, map_location=device))
    sar_model.eval()
    
    return ae_model, sar_model, ais_scaler, opt_thresh

@st.cache_data
def load_ais_data():
    csv_path = "Maritius_AOI_20200701_0731_full.csv"
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values(by=['mmsi', 'timestamp']).reset_index(drop=True)
    
    df['course_diff'] = df.groupby('mmsi')['course'].diff().fillna(0)
    df['rot_diff'] = df.groupby('mmsi')['rot'].diff().fillna(0)
    df['speed_diff'] = df.groupby('mmsi')['speed'].diff().fillna(0)
    df['lat_diff'] = df.groupby('mmsi')['latitude'].diff().fillna(0)
    df['long_diff'] = df.groupby('mmsi')['longitude'].diff().fillna(0)
    
    df['course_heading_drift'] = np.abs(df['course'] - df['heading'])
    drift_vals = np.where(df['course_heading_drift'] > 180, 
                          360 - df['course_heading_drift'], 
                          df['course_heading_drift'])
    df['course_heading_drift'] = pd.Series(drift_vals).fillna(0).values
    
    REEF_LAT, REEF_LON = -20.4411, 57.7525
    df['dist_to_reef'] = np.sqrt(
        ((df['latitude'] - REEF_LAT) * 111.0)**2 + 
        ((df['longitude'] - REEF_LON) * 111.0 * np.cos(np.radians(REEF_LAT)))**2
    )
    
    df['speed_rolling_std'] = df.groupby('mmsi')['speed'].rolling(5, min_periods=1).std().reset_index(0,drop=True).fillna(0)
    df['course_rolling_std'] = df.groupby('mmsi')['course'].rolling(5, min_periods=1).std().reset_index(0,drop=True).fillna(0)
    
    return df

# -------------------------------------------------------------
# App Layout & Navigation
# -------------------------------------------------------------
st.markdown('<div class="main-header">🛰️ Multimodal Maritime Oil Spill & AIS Surveillance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Vessel Anomaly Detection, Satellite SAR Oil Spill Segmentation & Legal Attribution (Smart India Hackathon)</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🚢 1. AIS Kinematic Anomaly Inspector", 
    "🛰️ 2. Satellite SAR Oil Spill Segmenter", 
    "🚨 3. Multimodal Attribution Command Center"
])

ae_model, sar_model, ais_scaler, opt_thresh = load_models()
df_ais = load_ais_data()

# -------------------------------------------------------------
# TAB 1: AIS Kinematic Anomaly Inspector
# -------------------------------------------------------------
with tab1:
    st.subheader("🚢 Real-Time Vessel Kinematic & Geospatial Anomaly Detection")
    st.write("Detects grounding risks, engine failure, erratic drift, and speed collapse using a Deep Trajectory Autoencoder.")
    
    if df_ais.empty:
        st.warning("AIS dataset not found.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        
        with col_ctrl1:
            mmsi_list = df_ais['mmsi'].value_counts().index.tolist()
            default_idx = mmsi_list.index(372711000) if 372711000 in mmsi_list else 0
            selected_mmsi = st.selectbox("Select Vessel MMSI:", mmsi_list, index=default_idx)
            
            vessel_df = df_ais[df_ais['mmsi'] == selected_mmsi].reset_index(drop=True)
            total_points = len(vessel_df)
            
            point_idx = st.slider("Telemetry Timestamp Step:", 0, total_points - 1, min(total_points - 1, 100))
            selected_row = vessel_df.iloc[point_idx]
            
        with col_ctrl2:
            st.markdown(f"**Vessel MMSI:** `{selected_mmsi}` | **Timestamp:** `{selected_row['timestamp']}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Speed (kn)", f"{selected_row['speed']:.1f}")
            c2.metric("Course (°)", f"{selected_row['course']:.1f}")
            c3.metric("Rate of Turn", f"{selected_row['rot']:.1f}")
            c4.metric("Dist to Reef", f"{selected_row['dist_to_reef']:.2f} km")
            
        feature_cols = [
            'speed', 'course', 'rot', 'course_heading_drift',
            'course_diff', 'rot_diff', 'speed_diff', 'lat_diff', 'long_diff',
            'dist_to_reef', 'speed_rolling_std', 'course_rolling_std'
        ]
        feat_vals = selected_row[feature_cols].values.reshape(1, -1)
        scaled_feat = ais_scaler.transform(feat_vals) if ais_scaler else feat_vals
        t_feat = torch.tensor(scaled_feat, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            recon = ae_model(t_feat)
            score = float(torch.mean((t_feat - recon)**2).cpu().item())
            
        is_anomaly = score > opt_thresh
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("### Anomaly Assessment")
            st.metric("Autoencoder Reconstruction Error", f"{score:.5f}", delta=f"Thresh: {opt_thresh:.5f}")
            if is_anomaly:
                st.markdown(f'<div class="alert-critical">⚠️ CRITICAL KINEMATIC ANOMALY DETECTED!<br>Score ({score:.4f}) > Threshold ({opt_thresh:.4f})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-normal">✅ NORMAL CRUISING PROFILE<br>Score ({score:.4f}) ≤ Threshold ({opt_thresh:.4f})</div>', unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("### Vessel Trajectory & Hazard Zone Map")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(vessel_df['longitude'], vessel_df['latitude'], color='gray', linestyle='--', alpha=0.6, label='Trajectory')
            ax.scatter(vessel_df['longitude'].iloc[:point_idx], vessel_df['latitude'].iloc[:point_idx], color='blue', s=10, alpha=0.5)
            ax.scatter(selected_row['longitude'], selected_row['latitude'], color='red' if is_anomaly else 'green', s=120, zorder=5, label='Current Ping')
            ax.scatter(57.7525, -20.4411, color='orange', marker='*', s=200, label='Pointe d\'Esny Reef')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Vessel {selected_mmsi} Track (Mauritius AOI)")
            ax.legend(loc='upper left')
            st.pyplot(fig)

# -------------------------------------------------------------
# TAB 2: SAR Satellite Oil Spill Segmenter
# -------------------------------------------------------------
with tab2:
    st.subheader("🛰️ Sentinel-1 SAR Satellite Oil Spill Semantic Segmentation")
    st.write("Segments SAR scenes into 5 classes with class-weighted Focal-Dice U-Net to distinguish crude oil from look-alikes.")
    
    test_img_paths = sorted(glob.glob("Oil Spill Detection Dataset/test/images/*.jpg"))
    test_lbl_paths = sorted(glob.glob("Oil Spill Detection Dataset/test/labels_1D/*.png"))
    
    if not test_img_paths:
        st.warning("SAR test dataset not found.")
    else:
        sar_col1, sar_col2 = st.columns([1, 2])
        
        with sar_col1:
            selected_scene_idx = st.selectbox(
                "Select Test SAR Scene:", 
                range(len(test_img_paths)), 
                format_func=lambda x: f"Scene #{x+1:03d} ({os.path.basename(test_img_paths[x])})"
            )
            
            chosen_img_path = test_img_paths[selected_scene_idx]
            chosen_lbl_path = test_lbl_paths[selected_scene_idx]
            
            raw_img = Image.open(chosen_img_path).convert("RGB").resize((256, 256))
            img_tensor = torch.tensor(np.array(raw_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            
            gt_1d = np.array(Image.open(chosen_lbl_path).resize((256, 256), Image.NEAREST), dtype=np.int64)
            
            with torch.no_grad():
                pred_logits = sar_model(img_tensor)
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
                
            total_px = pred_mask.size
            oil_px = np.sum(pred_mask == 1)
            lookalike_px = np.sum(pred_mask == 2)
            ship_px = np.sum(pred_mask == 3)
            land_px = np.sum(pred_mask == 4)
            
            st.markdown("### Scene Composition")
            st.write(f"• **Oil Spill:** {oil_px:,} px ({oil_px/total_px*100:.2f}%)")
            st.write(f"• **Look-alike:** {lookalike_px:,} px ({lookalike_px/total_px*100:.2f}%)")
            st.write(f"• **Ship Target:** {ship_px:,} px ({ship_px/total_px*100:.2f}%)")
            st.write(f"• **Estimated Slick Area:** `{oil_px * 0.01:.2f} km²`")
            
        with sar_col2:
            fig_sar, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(raw_img)
            axes[0].set_title("Input Sentinel-1 SAR")
            axes[0].axis('off')
            
            axes[1].imshow(mask_to_rgb(gt_1d))
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            axes[2].imshow(mask_to_rgb(pred_mask))
            axes[2].set_title("U-Net Predicted Mask")
            axes[2].axis('off')
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='gray', label='Sea (0)'),
                Patch(facecolor='cyan', label='Oil Spill (1)'),
                Patch(facecolor='red', label='Look-alike (2)'),
                Patch(facecolor='#994C00', label='Ship (3)'),
                Patch(facecolor='#009900', label='Land (4)')
            ]
            fig_sar.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
            st.pyplot(fig_sar)

# -------------------------------------------------------------
# TAB 3: Multimodal Early Warning & Attribution Command Center
# -------------------------------------------------------------
with tab3:
    st.subheader("🚨 Multimodal AIS-SAR Fusion & Legal Attribution Center")
    st.write("Correlates vessel kinematic anomaly triggers directly with satellite SAR imagery to confirm spills and establish legal responsibility.")
    
    multi_col1, multi_col2 = st.columns(2)
    
    with multi_col1:
        st.markdown("### Step 1: Vessel Telemetry Stream")
        input_mmsi = st.number_input("Target Vessel MMSI:", value=372711000)
        input_speed = st.slider("Vessel Speed (knots):", 0.0, 25.0, 0.0)
        input_drift = st.slider("Course-Heading Drift (°):", 0.0, 90.0, 54.0)
        input_rot = st.slider("Rate of Turn (ROT):", -128.0, 128.0, -128.0)
        input_dist_reef = st.slider("Distance to Reef (km):", 0.0, 20.0, 0.05)
        
    with multi_col2:
        st.markdown("### Step 2: Automated Satellite SAR Confirmation")
        sar_test_choice = st.selectbox(
            "Tasked Satellite SAR Scene:", 
            range(min(10, len(test_img_paths))), 
            format_func=lambda x: f"Tasked AOI Scene #{x+1:02d}"
        )
        
        feat_vector = np.array([[
            input_speed, 292.0, input_rot, input_drift,
            45.0, -120.0, -11.0, 0.0001, 0.0001,
            input_dist_reef, 4.5, 22.0
        ]])
        scaled_vec = ais_scaler.transform(feat_vector) if ais_scaler else feat_vector
        t_vec = torch.tensor(scaled_vec, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            recon_vec = ae_model(t_vec)
            anomaly_val = float(torch.mean((t_vec - recon_vec)**2).cpu().item())
            
        is_ais_alarm = anomaly_val > opt_thresh
        
        tasked_sar_path = test_img_paths[sar_test_choice]
        tasked_img = Image.open(tasked_sar_path).convert("RGB").resize((256, 256))
        tasked_tensor = torch.tensor(np.array(tasked_img, dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            tasked_pred = torch.argmax(sar_model(tasked_tensor), dim=1).squeeze(0).cpu().numpy()
            
        spill_detected = int(np.sum(tasked_pred == 1)) > 50
        est_area = round(np.sum(tasked_pred == 1) * 0.01, 2)
        
    st.markdown("---")
    st.markdown("### 📋 Official Maritime Incident & Attribution Dossier")
    
    dossier_col1, dossier_col2 = st.columns([1, 1])
    
    with dossier_col1:
        st.json({
            "Incident Timestamp": str(datetime.utcnow()),
            "Target Vessel MMSI": int(input_mmsi),
            "AIS Kinematic Anomaly Alert": bool(is_ais_alarm),
            "Reconstruction Error Score": round(anomaly_val, 4),
            "Decision Threshold": round(opt_thresh, 4),
            "Satellite SAR Oil Slick Confirmed": bool(spill_detected),
            "Estimated Slick Size": f"{est_area} km²",
            "Legal Responsibility Status": "CONFIRMED OFFENDER: ATTRIBUTED TO VESSEL MMSI" if (is_ais_alarm and spill_detected)
                                         else "POTENTIAL HAZARD: MONITORING VESSEL" if is_ais_alarm
                                         else "NORMAL MARITIME OPERATION"
        })
        
    with dossier_col2:
        if is_ais_alarm and spill_detected:
            st.error(f"🚨 **CRITICAL MARITIME DISASTER DETECTED**\nVessel MMSI `{input_mmsi}` has experienced erratic kinematic failure near shallow reefs and satellite SAR confirms active oil spill spreading ({est_area} km²). Attribution is locked.")
        elif is_ais_alarm:
            st.warning(f"⚠️ **KINEMATIC ANOMALY DETECTED**\nVessel MMSI `{input_mmsi}` is behaving abnormally, but no oil spill is currently detected in the satellite radar scene.")
        else:
            st.success("✅ **ALL SYSTEMS NORMAL**\nVessel kinematics and satellite radar reflect standard safe operations.")
