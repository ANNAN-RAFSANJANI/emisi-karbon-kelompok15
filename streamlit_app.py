import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG & STYLING
# =========================
st.set_page_config(page_title="🌍 Informer Forecast Dashboard", layout="wide", page_icon="🌎")

st.markdown("""
<style>
/* Page background and glass containers */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg,#071224 0%, #092640 50%, #0f3b5a 100%);
}
.block-container {
  padding: 1.2rem 1.6rem;
  border-radius: 14px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(8,12,20,0.92);
  color: #e6f7f2;
  border-right: 1px solid rgba(255,255,255,0.04);
}

/* Title styles */
h1, h2, h3 {
  color: #dffcf6 !important;
  text-shadow: 0 2px 10px rgba(0,200,180,0.06);
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#00E3CC,#0066FF) !important;
  color: white !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  font-weight: 600;
}

/* Small helper text */
.small-muted { color: #b8c6c2; font-size:0.9rem; }

/* Plotly text */
.plotly-graph-div .modebar, .plotly-graph-div .main-svg { color: #eafaf5 !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG: files & model
# =========================
# Accept Excel filename variations; find a file that starts with the base name
EXCEL_BASENAME = "National_LandUseChange_Carbon_Emissions_Clean"
def find_excel_file():
    for f in os.listdir("."):
        if f.startswith(EXCEL_BASENAME) and f.lower().endswith((".xlsx", ".xls")):
            return f
    return None

EXCEL_FILE = find_excel_file()
INPUT_LEN, PRED_LEN, LABEL_LEN = 44, 50, 15
PICKLES = {
    "BLUE": "informer_model_BLUE_cpu.pkl",
    "H&C2023": "informer_model_H&C2023_cpu.pkl",
    "OSCAR": "informer_model_OSCAR_cpu.pkl",
    "LUCE": "informer_model_LUCE_cpu.pkl"
}

# add Informer path if exists
HERE = os.path.dirname(__file__)
INFO_PATH = os.path.join(HERE, "Informer2020")
if INFO_PATH not in sys.path:
    sys.path.append(INFO_PATH)

# try import Informer
try:
    from Informer2020.models.model import Informer
except Exception:
    Informer = None

# =========================
# UTILITIES
# =========================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_state_dict(sd: dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

@st.cache_data
def load_reference_sheet():
    """Load a reference sheet (first available sheet) to build default map & countries."""
    if EXCEL_FILE is None:
        return None, None
    try:
        xls = pd.ExcelFile(EXCEL_FILE)
        # prefer sheet that matches a PICKLES key, else first sheet
        sheet_candidates = [s for s in xls.sheet_names if s in PICKLES.keys()]
        sheet_name = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
        ref = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        return ref, sheet_name
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None, None

@st.cache_resource
def load_model_and_scaler(pickle_path, enc_in):
    if Informer is None:
        raise RuntimeError("Informer model class not available (Informer2020 import failed).")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        saved = pickle.load(f)
    scaler = saved.get("scaler", None)
    device = get_device()
    model = Informer(enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                     seq_len=INPUT_LEN, label_len=LABEL_LEN, out_len=PRED_LEN,
                     d_model=256, n_heads=8, e_layers=2, d_layers=1,
                     dropout=0.1, attn='prob', embed='timeF', freq='a',
                     activation='gelu', output_attention=False,
                     distil=True, mix=True, device=device)
    state = saved.get("model_state_dict", saved)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, scaler, device

@st.cache_data
def predict_for_model(model_name):
    if EXCEL_FILE is None:
        raise FileNotFoundError("Excel file not found in working directory.")
    df = pd.read_excel(EXCEL_FILE, sheet_name=model_name).dropna()
    years = df.iloc[:, 0].values
    countries = df.columns[1:].tolist()
    data_values = df.iloc[:, 1:].values  # shape (T, features)
    model, scaler, device = load_model_and_scaler(PICKLES[model_name], enc_in=data_values.shape[1])
    if scaler is None:
        raise RuntimeError("Scaler not found in pickle. Add 'scaler' to pickle.")

    data_scaled = scaler.transform(data_values)  # (T, enc_in)
    x_enc = torch.tensor(data_scaled[-INPUT_LEN:], dtype=torch.float32).unsqueeze(0).to(device)
    x_mark_enc = torch.arange(0, INPUT_LEN, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
    x_mark_dec = torch.arange(INPUT_LEN, INPUT_LEN + PRED_LEN, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
    zeros = torch.zeros(1, PRED_LEN - LABEL_LEN, x_enc.shape[-1], dtype=torch.float32).to(device)
    x_dec = torch.cat([x_enc[:, -LABEL_LEN:, :], zeros], dim=1).to(device)

    # ensure float and same device
    x_enc = x_enc.float(); x_dec = x_dec.float()
    x_mark_enc = x_mark_enc.float().to(device); x_mark_dec = x_mark_dec.float().to(device)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    preds = out.squeeze(0).cpu().numpy()  # (pred_len, enc_in)
    preds_inv = scaler.inverse_transform(preds)
    future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + 1 + preds_inv.shape[0])
    df_future = pd.DataFrame(preds_inv, columns=countries)
    df_future.insert(0, "Year", future_years)
    return df_future

def smooth_prediction(ser_hist, ser_pred, steps=5):
    ser_pred = ser_pred.copy()
    for i in range(min(steps, len(ser_pred))):
        alpha = (i + 1) / (steps + 1)
        ser_pred[i] = (1 - alpha) * ser_hist[-1] + alpha * ser_pred[i]
    return ser_pred

# =========================
# LAYOUT: header + sidebar
# =========================
st.markdown("<h1 style='text-align:center'>🌍 INFORMER — Carbon Emission Forecast</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar controls & reference
st.sidebar.header("⚙ Controls")

# Load available sheets
if EXCEL_FILE is None:
    st.sidebar.error("Excel file not found. Put the Excel file next to app.py and retry.")
    st.stop()

try:
    xls = pd.ExcelFile(EXCEL_FILE)
    available_sheets = [s for s in xls.sheet_names if s in PICKLES.keys()]
    if not available_sheets:
        available_sheets = xls.sheet_names
    
    # Select reference sheet
    ref_sheet_name = st.sidebar.selectbox("Pilih Reference Sheet", available_sheets, index=0)
    ref_df = pd.read_excel(EXCEL_FILE, sheet_name=ref_sheet_name)
    
    years_hist = ref_df.iloc[:, 0].values
    countries = ref_df.columns[1:].tolist()
    
    default_year = int(years_hist[-1])
    year_for_map = st.sidebar.slider("Tahun Data Peta Global", int(years_hist[0]), int(years_hist[-1]), default_year)
    mode = st.sidebar.radio("Fitur", ["Prediksi Per Negara", "Perbandingan Model"])
    smooth_val = st.sidebar.slider("Smoothing steps (blend)", 0, 10, 5)
except Exception as e:
    st.sidebar.error(f"Error loading Excel file: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("Model pickles")
for k, p in PICKLES.items():
    st.sidebar.markdown(f"- *{k}* → {p}")
st.sidebar.caption("Kelompok 15")

# =========================
# Custom color scale
# =========================
# Custom color scale from light yellow to dark red
custom_color_scale = [
    [0.0, '#FFFFCC'],   # Very light yellow
    [0.14, '#FFEDA0'],  # Light yellow
    [0.29, '#FED976'],  # Yellow
    [0.43, '#FEB24C'],  # Light orange
    [0.57, '#FD8D3C'],  # Orange
    [0.71, '#FC4E2A'],  # Red-orange
    [0.86, '#E31A1C'],  # Red
    [1.0, '#B10026']    # Dark red
]

# =========================
# Default GLOBAL MAP
# =========================
st.markdown("## 🌐 Global Emission")
try:
    df_hist = ref_df.copy()
    df_map_default = df_hist[df_hist.iloc[:, 0] == year_for_map].melt(id_vars=df_hist.columns[0], var_name="Country", value_name="Emissions")
    df_map_default.rename(columns={df_hist.columns[0]: "Year"}, inplace=True)
except Exception as e:
    st.error(f"Failed preparing default map: {e}")
    df_map_default = pd.DataFrame(columns=["Year", "Country", "Emissions"])

fig_default_map = px.choropleth(
    df_map_default,
    locations="Country",
    locationmode="country names",
    color="Emissions",
    hover_name="Country",
    color_continuous_scale=custom_color_scale,
    title=f"Global Emissions — {year_for_map}"
)
fig_default_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_default_map, use_container_width=True)

# Top 10 Countries with Highest Emissions
st.markdown("### 🔝 Top 10 Negara dengan Emisi Tertinggi")
if not df_map_default.empty:
    top10 = df_map_default.nlargest(10, 'Emissions')[['Country', 'Emissions']].reset_index(drop=True)
    top10.index = top10.index + 1
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig_top10 = px.bar(
            top10, 
            x='Emissions', 
            y='Country', 
            orientation='h',
            color='Emissions',
            color_continuous_scale=custom_color_scale,
            title=f"Top 10 Emisi Karbon — {year_for_map}",
            labels={'Emissions': 'Emisi Karbon', 'Country': 'Negara'}
        )
        fig_top10.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_top10, use_container_width=True)
    
    with col_table:
        st.markdown("**Tabel Ranking**")
        top10_display = top10.copy()
        top10_display['Emissions'] = top10_display['Emissions'].apply(lambda x: f"{x:.2f}")
        st.dataframe(
            top10_display,
            hide_index=False,
            use_container_width=True,
            height=400
        )

# =========================
# PREDICT (Single) or COMPARE MODE
# =========================
if mode == "Prediksi Per Negara":
    st.header("Prediski Per Negara")
    col1, col2 = st.columns([2, 1])
        
    with col1:
        st.markdown("*Peta Dunia*")
        selected_model = st.selectbox("Pilih Model", list(PICKLES.keys()))
        run_btn = st.button("🚀 Jalankan Prediksi")
        st.caption("Pastikan file pickle model tersedia. Berjalan di CPU.")

    if run_btn:
        # run prediction
        try:
            with st.spinner(f"Running prediction for {selected_model}..."):
                df_future = predict_for_model(selected_model)
                st.session_state[f"df_future_{selected_model}"] = df_future
            st.success("Prediction finished")
        except Exception as e:
            st.exception(e)

    key = f"df_future_{selected_model}"
    if key in st.session_state:
        df_future = st.session_state[key]

        # calendar slider across historical+predicted
        combined_years = np.concatenate([ref_df.iloc[:, 0].values, df_future["Year"].values])
        combined_years = np.unique(combined_years)
        year_choice = st.select_slider("Pilih Tahun (historical + predicted)", options=combined_years, value=int(df_future["Year"].iloc[0]))

        # build map data depending on year_choice
        if int(year_choice) in list(df_future["Year"].astype(int).values):
            df_map = df_future[df_future["Year"] == int(year_choice)].melt(id_vars="Year", var_name="Country", value_name="Emissions")
        else:
            hist_sheet = pd.read_excel(EXCEL_FILE, sheet_name=selected_model).dropna()
            df_map = hist_sheet[hist_sheet.iloc[:, 0] == int(year_choice)].melt(id_vars=hist_sheet.columns[0], var_name="Country", value_name="Emissions")
            df_map.rename(columns={hist_sheet.columns[0]: "Year"}, inplace=True)

        fig = px.choropleth(df_map, locations="Country", locationmode="country names",
                            color="Emissions", hover_name="Country",
                            color_continuous_scale=custom_color_scale,
                            title=f"{selected_model} — Emissions {int(year_choice)}")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 Countries for selected year
        if not df_map.empty:
            st.markdown(f"### 🔝 Top 10 Negara dengan Emisi Tertinggi — Tahun {int(year_choice)}")
            top10_pred = df_map.nlargest(10, 'Emissions')[['Country', 'Emissions']].reset_index(drop=True)
            top10_pred.index = top10_pred.index + 1
            
            col_chart_pred, col_table_pred = st.columns([2, 1])
            
            with col_chart_pred:
                fig_top10_pred = px.bar(
                    top10_pred,
                    x='Emissions',
                    y='Country',
                    orientation='h',
                    color='Emissions',
                    color_continuous_scale=custom_color_scale,
                    title=f"Top 10 Emisi Karbon — {selected_model} ({int(year_choice)})",
                    labels={'Emissions': 'Emisi Karbon', 'Country': 'Negara'}
                )
                fig_top10_pred.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_top10_pred, use_container_width=True)
            
            with col_table_pred:
                st.markdown("**Tabel Ranking**")
                top10_pred_display = top10_pred.copy()
                top10_pred_display['Emissions'] = top10_pred_display['Emissions'].apply(lambda x: f"{x:.2f}")
                st.dataframe(
                    top10_pred_display,
                    hide_index=False,
                    use_container_width=True,
                    height=400
                )

        # =========================
        # NEW: Yearly Global Emission Charts
        # =========================
        st.markdown("---")
        st.markdown("### Analisis Emisi Global Per Tahun")
        
        # Load historical data
        hist_sheet = pd.read_excel(EXCEL_FILE, sheet_name=selected_model).dropna()
        years_hist_all = hist_sheet.iloc[:, 0].values
        
        # Calculate total emissions per year (historical)
        total_hist = hist_sheet.iloc[:, 1:].sum(axis=1).values
        
        # Calculate total emissions per year (predicted)
        total_pred = df_future.iloc[:, 1:].sum(axis=1).values
        years_pred_all = df_future["Year"].values
        
        # Smooth the prediction
        total_pred_smooth = smooth_prediction(total_hist, total_pred, steps=smooth_val)
        
        # Create combined dataframe for plotting
        df_yearly_total = pd.DataFrame({
            "Year": np.concatenate([years_hist_all, years_pred_all]),
            "Total Emissions": np.concatenate([total_hist, total_pred_smooth]),
            "Type": ["Historical"] * len(years_hist_all) + ["Predicted"] * len(years_pred_all)
        })
        
        # Total Global Emissions Over Time
        fig_yearly_total = go.Figure()
        
        # Historical line
        hist_data = df_yearly_total[df_yearly_total["Type"] == "Historical"]
        fig_yearly_total.add_trace(go.Scatter(
            x=hist_data["Year"],
            y=hist_data["Total Emissions"],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#9be7d2', width=3),
            marker=dict(size=6)
        ))
        
        # Predicted line
        pred_data = df_yearly_total[df_yearly_total["Type"] == "Predicted"]
        fig_yearly_total.add_trace(go.Scatter(
            x=pred_data["Year"],
            y=pred_data["Total Emissions"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7fa1', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig_yearly_total.update_layout(
            title=f"Total Emisi Global Per Tahun — {selected_model}",
            xaxis_title="Tahun",
            yaxis_title="Total Emisi Karbon",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode='x unified',
            height=450
        )
        
        st.plotly_chart(fig_yearly_total, use_container_width=True)

        # =========================
        # STATISTIK PREDIKSI
        # =========================
        st.markdown("---")
        st.markdown("### Statistik Hasil Prediksi")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        # Global statistics
        total_pred_all = df_future.iloc[:, 1:].sum(axis=1).values
        avg_yearly_pred = np.mean(total_pred_all)
        max_yearly_pred = np.max(total_pred_all)
        min_yearly_pred = np.min(total_pred_all)
        
        with col_stat1:
            st.metric(
                label="📊 Rata-rata Emisi Tahunan (Prediksi)",
                value=f"{avg_yearly_pred:,.2f}",
                delta=f"{((avg_yearly_pred - total_hist[-1]) / total_hist[-1] * 100):.2f}%"
            )
        
        with col_stat2:
            st.metric(
                label="📈 Emisi Maksimum (Prediksi)",
                value=f"{max_yearly_pred:,.2f}",
                delta=f"Tahun {df_future.loc[df_future.iloc[:, 1:].sum(axis=1).idxmax(), 'Year']:.0f}"
            )
        
        with col_stat3:
            st.metric(
                label="📉 Emisi Minimum (Prediksi)",
                value=f"{min_yearly_pred:,.2f}",
                delta=f"Tahun {df_future.loc[df_future.iloc[:, 1:].sum(axis=1).idxmin(), 'Year']:.0f}"
            )
        
        # Tren Prediksi
        st.markdown("#### Ringkasan Tren Prediksi")
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            # Perubahan total emisi
            first_pred_year = total_pred_all[0]
            last_pred_year = total_pred_all[-1]
            change_pct = ((last_pred_year - first_pred_year) / first_pred_year) * 100
            
            st.info(f"""
            **Perubahan Total Emisi:**
            - Tahun Awal Prediksi: {first_pred_year:,.2f}
            - Tahun Akhir Prediksi: {last_pred_year:,.2f}
            - Perubahan: {change_pct:+.2f}%
            """)
        
        with col_trend2:
            # Growth rate
            years_range = len(total_pred_all)
            avg_growth_rate = ((last_pred_year / first_pred_year) ** (1/years_range) - 1) * 100
            
            st.info(f"""
            **Tingkat Pertumbuhan:**
            - Rata-rata Pertumbuhan Tahunan: {avg_growth_rate:+.2f}%
            - Periode Prediksi: {years_range} tahun
            - Total Emisi Kumulatif: {np.sum(total_pred_all):,.2f}
            """)

        # Time series for a selected country
        st.markdown("---")
        st.markdown("### Analisis Per Negara")
        chosen_country = st.selectbox("Pilih Negara untuk Time Series", countries)
        hist = pd.read_excel(EXCEL_FILE, sheet_name=selected_model).dropna()
        ser_hist = hist[chosen_country].values
        years_hist_local = hist.iloc[:, 0].values
        ser_pred_local = df_future[chosen_country].values
        years_pred_local = df_future["Year"].values
        ser_pred_smooth = smooth_prediction(ser_hist, ser_pred_local, steps=smooth_val)
        
        # Statistik per negara
        col_country1, col_country2, col_country3, col_country4 = st.columns(4)
        
        with col_country1:
            st.metric(
                label=f"Emisi Terakhir (Historis)",
                value=f"{ser_hist[-1]:.2f}"
            )
        
        with col_country2:
            st.metric(
                label=f"Rata-rata Prediksi",
                value=f"{np.mean(ser_pred_smooth):.2f}",
                delta=f"{((np.mean(ser_pred_smooth) - ser_hist[-1]) / ser_hist[-1] * 100):+.2f}%"
            )
        
        with col_country3:
            st.metric(
                label=f"Emisi Maksimum (Prediksi)",
                value=f"{np.max(ser_pred_smooth):.2f}"
            )
        
        with col_country4:
            st.metric(
                label=f"Emisi Minimum (Prediksi)",
                value=f"{np.min(ser_pred_smooth):.2f}"
            )
        
        df_ts = pd.DataFrame({
            "Year": np.concatenate([years_hist_local, years_pred_local]),
            "Value": np.concatenate([ser_hist, ser_pred_smooth]),
            "Type": ["Historical"] * len(years_hist_local) + ["Predicted"] * len(years_pred_local)
        })
        fig_ts = px.line(df_ts, x="Year", y="Value", color="Type", markers=True,
                         color_discrete_map={"Historical": "#9be7d2", "Predicted": "#ff7fa1"},
                         title=f"{chosen_country} — {selected_model}")
        fig_ts.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ts, use_container_width=True)

else:  # Compare Models
    st.header("⚔ Perbandingan Model")
    run_all = st.button("🚀 Jalankan Semua Prediksi")
    if run_all:
        st.session_state["df_future_all"] = {}
        errors = {}
        for m in PICKLES.keys():
            try:
                st.session_state["df_future_all"][m] = predict_for_model(m)
            except Exception as e:
                errors[m] = str(e)
        if errors:
            st.error("Some models failed: " + "; ".join([f"{k}: {v}" for k, v in errors.items()]))
        else:
            st.success("All models predicted successfully")

    if "df_future_all" in st.session_state and st.session_state["df_future_all"]:
        # =========================
        # GLOBAL COMPARISON
        # =========================
        st.markdown("### Perbandingan Emisi Global")
        st.markdown("Analisis total emisi global dari semua model")
        
        # Prepare global comparison data
        global_comparison_data = []
        model_colors = {
            "BLUE": "#00E3CC",
            "H&C2023": "#FF6B9D",
            "OSCAR": "#FFA07A",
            "LUCE": "#9370DB"
        }
        
        for model_name, df_future in st.session_state["df_future_all"].items():
            # Historical data
            hist = pd.read_excel(EXCEL_FILE, sheet_name=model_name).dropna()
            years_hist_local = hist.iloc[:, 0].values
            total_hist = hist.iloc[:, 1:].sum(axis=1).values
            
            # Predicted data
            total_pred = df_future.iloc[:, 1:].sum(axis=1).values
            total_pred_smooth = smooth_prediction(total_hist, total_pred, steps=smooth_val)
            years_pred_local = df_future["Year"].values
            
            # Combine
            for y, v in zip(years_hist_local, total_hist):
                global_comparison_data.append({
                    "Year": y,
                    "Total Emissions": v,
                    "Model": model_name,
                    "Type": "Historical"
                })
            for y, v in zip(years_pred_local, total_pred_smooth):
                global_comparison_data.append({
                    "Year": y,
                    "Total Emissions": v,
                    "Model": model_name,
                    "Type": "Predicted"
                })
        
        df_global_comp = pd.DataFrame(global_comparison_data)
        
        # Chart: Global Emissions Comparison
        fig_global = go.Figure()
        
        for model_name in PICKLES.keys():
            model_data = df_global_comp[df_global_comp["Model"] == model_name]
            hist_data = model_data[model_data["Type"] == "Historical"]
            pred_data = model_data[model_data["Type"] == "Predicted"]
            
            # Historical line
            fig_global.add_trace(go.Scatter(
                x=hist_data["Year"],
                y=hist_data["Total Emissions"],
                mode='lines',
                name=f'{model_name} (Historical)',
                line=dict(color=model_colors.get(model_name, "#CCCCCC"), width=2),
                showlegend=True
            ))
            
            # Predicted line
            fig_global.add_trace(go.Scatter(
                x=pred_data["Year"],
                y=pred_data["Total Emissions"],
                mode='lines',
                name=f'{model_name} (Predicted)',
                line=dict(color=model_colors.get(model_name, "#CCCCCC"), width=2, dash='dash'),
                showlegend=True
            ))
        
        fig_global.update_layout(
            title="Perbandingan Total Emisi Global — Semua Model",
            xaxis_title="Tahun",
            yaxis_title="Total Emisi Global",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        st.plotly_chart(fig_global, use_container_width=True)
        
        # Statistics comparison table
        st.markdown("#### Statistik Perbandingan Model (Prediksi)")
        
        stats_data = []
        for model_name, df_future in st.session_state["df_future_all"].items():
            hist = pd.read_excel(EXCEL_FILE, sheet_name=model_name).dropna()
            total_hist = hist.iloc[:, 1:].sum(axis=1).values
            total_pred = df_future.iloc[:, 1:].sum(axis=1).values
            total_pred_smooth = smooth_prediction(total_hist, total_pred, steps=smooth_val)
            
            stats_data.append({
                "Model": model_name,
                "Rata-rata": f"{np.mean(total_pred_smooth):,.2f}",
                "Maksimum": f"{np.max(total_pred_smooth):,.2f}",
                "Minimum": f"{np.min(total_pred_smooth):,.2f}",
                "Total Kumulatif": f"{np.sum(total_pred_smooth):,.2f}",
                "Perubahan (%)": f"{((total_pred_smooth[-1] - total_pred_smooth[0]) / total_pred_smooth[0] * 100):+.2f}%"
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Box plot comparison
        st.markdown("#### Distribusi Prediksi Model")
        
        box_data = []
        for model_name, df_future in st.session_state["df_future_all"].items():
            hist = pd.read_excel(EXCEL_FILE, sheet_name=model_name).dropna()
            total_hist = hist.iloc[:, 1:].sum(axis=1).values
            total_pred = df_future.iloc[:, 1:].sum(axis=1).values
            total_pred_smooth = smooth_prediction(total_hist, total_pred, steps=smooth_val)
            
            for val in total_pred_smooth:
                box_data.append({
                    "Model": model_name,
                    "Emisi": val
                })
        
        df_box = pd.DataFrame(box_data)
        fig_box = px.box(
            df_box,
            x="Model",
            y="Emisi",
            color="Model",
            color_discrete_map=model_colors,
            title="Distribusi Emisi Global — Perbandingan Model"
        )
        fig_box.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        
        # =========================
        # PER COUNTRY COMPARISON
        # =========================
        st.markdown("### Perbandingan Per Negara")
        chosen_countries = st.multiselect("Pilih Negara untuk Perbandingan", countries, default=countries[:1])
        
        if chosen_countries:
            fig_comp = go.Figure()
            
            for model_name, df_future in st.session_state["df_future_all"].items():
                hist = pd.read_excel(EXCEL_FILE, sheet_name=model_name).dropna()
                years_hist_local = hist.iloc[:, 0].values
                
                for country in chosen_countries:
                    ser_hist = hist[country].values
                    ser_pred = df_future[country].values
                    ser_pred_smooth = smooth_prediction(ser_hist, ser_pred, steps=smooth_val)
                    years_pred_local = df_future["Year"].values
                    
                    # Historical
                    fig_comp.add_trace(go.Scatter(
                        x=years_hist_local,
                        y=ser_hist,
                        mode="lines",
                        name=f"{model_name} — {country} (Hist)",
                        line=dict(width=2),
                        showlegend=True
                    ))
                    
                    # Predicted
                    fig_comp.add_trace(go.Scatter(
                        x=years_pred_local,
                        y=ser_pred_smooth,
                        mode="lines",
                        name=f"{model_name} — {country} (Pred)",
                        line=dict(width=2, dash='dash'),
                        showlegend=True
                    ))
            
            fig_comp.update_layout(
                title="Perbandingan Emisi Per Negara — Semua Model",
                xaxis_title="Tahun",
                yaxis_title="Emisi",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Pilih minimal satu negara untuk melihat perbandingan")
    else:
        st.info("Belum ada data perbandingan. Tekan 'Jalankan Semua Prediksi' untuk menjalankan semua model.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<div style='text-align:center; color:#b7dcd0'>© 2025 Kelompok 15 — Informer Forecast Dashboard</div>", unsafe_allow_html=True)