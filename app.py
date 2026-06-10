import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Isolation Forest – Anomaly Detection",
    page_icon="🛍️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-title { color: #a6adc8; font-size: 13px; margin-bottom: 4px; }
  .metric-value { color: #cdd6f4; font-size: 32px; font-weight: 700; }
  .metric-sub   { color: #6c7086; font-size: 12px; margin-top: 2px; }
  .anomaly  { color: #f38ba8; }
  .normal   { color: #a6e3a1; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛍️ Isolation Forest – Anomaly Detection")
st.caption("Mall Customers Dataset · Unsupervised anomaly detection using Isolation Forest")
st.divider()

# ── Sidebar – Data Upload & Model Config ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded = st.file_uploader("Upload CSV (optional – uses built-in data by default)", type=["csv"])

    st.subheader("Model Parameters")
    contamination = st.slider(
        "Contamination (expected anomaly fraction)",
        min_value=0.01, max_value=0.30, value=0.05, step=0.01,
        help="Proportion of observations expected to be anomalies."
    )
    n_estimators = st.slider(
        "Number of Trees",
        min_value=50, max_value=500, value=100, step=50
    )
    max_samples = st.select_slider(
        "Max Samples",
        options=["auto", 64, 128, 256, 512],
        value="auto"
    )
    random_state = st.number_input("Random State", value=42, step=1)

    st.subheader("Features")
    feature_options = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    selected_features = st.multiselect(
        "Select features for the model",
        options=feature_options,
        default=feature_options,
    )

    run_btn = st.button("🚀 Run Detection", use_container_width=True, type="primary")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_default():
    """Return the built-in Mall Customers dataset as a DataFrame."""
    # Embedded dataset (200 rows – identical to Mall_Customers.csv)
    import io, requests, pathlib, os
    # Try to read from common upload paths first
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.path.join(script_dir, "Mall_Customers.csv"),
        "Mall_Customers.csv",
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    # Fallback: minimal synthetic placeholder so the app still runs
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 70, n),
        "Annual Income (k$)": rng.integers(15, 137, n),
        "Spending Score (1-100)": rng.integers(1, 99, n),
    })

if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = load_default()

# ── Validate features ─────────────────────────────────────────────────────────
if not selected_features:
    st.warning("⚠️ Please select at least one feature in the sidebar.")
    st.stop()

missing = [f for f in selected_features if f not in df_raw.columns]
if missing:
    st.error(f"Selected features not found in data: {missing}")
    st.stop()

# ── Raw Data Preview ──────────────────────────────────────────────────────────
with st.expander("📋 Raw Data Preview", expanded=False):
    st.dataframe(df_raw, use_container_width=True, height=250)
    st.caption(f"{df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ── Run model on button press (or first load) ─────────────────────────────────
if "results" not in st.session_state or run_btn:
    X = df_raw[selected_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=max_samples if max_samples != "auto" else "auto",
        contamination=contamination,
        random_state=int(random_state),
    )
    preds = model.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)   # lower → more anomalous

    df_out = df_raw.copy()
    df_out["Anomaly Score"] = scores
    df_out["Prediction"]    = np.where(preds == -1, "Anomaly", "Normal")

    st.session_state["results"]  = df_out
    st.session_state["model"]    = model
    st.session_state["features"] = selected_features

df_out = st.session_state["results"]
n_anomaly = (df_out["Prediction"] == "Anomaly").sum()
n_normal  = (df_out["Prediction"] == "Normal").sum()
pct       = n_anomaly / len(df_out) * 100

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{len(df_out)}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Normal</div>
        <div class="metric-value normal">{n_normal}</div>
        <div class="metric-sub">{100 - pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Anomalies Detected</div>
        <div class="metric-value anomaly">{n_anomaly}</div>
        <div class="metric-sub">{pct:.1f}% of dataset</div>
    </div>""", unsafe_allow_html=True)
with c4:
    avg_score_anomaly = df_out.loc[df_out["Prediction"] == "Anomaly", "Anomaly Score"].mean()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Avg Anomaly Score</div>
        <div class="metric-value anomaly">{avg_score_anomaly:.4f}</div>
        <div class="metric-sub">lower = more anomalous</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {"Normal": "#a6e3a1", "Anomaly": "#f38ba8"}

# ── Plots ─────────────────────────────────────────────────────────────────────
feats = st.session_state["features"]
plot_bg  = "#1e1e2e"
plot_fg  = "#cdd6f4"
grid_col = "#313244"

def style_ax(ax):
    ax.set_facecolor(plot_bg)
    ax.tick_params(colors=plot_fg, labelsize=9)
    ax.xaxis.label.set_color(plot_fg)
    ax.yaxis.label.set_color(plot_fg)
    ax.title.set_color(plot_fg)
    for sp in ax.spines.values():
        sp.set_color(grid_col)
    ax.grid(color=grid_col, linestyle="--", linewidth=0.5, alpha=0.6)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Scatter Plots",
    "📉 Anomaly Score Distribution",
    "🔥 Feature Heatmap",
    "📋 Detailed Results",
])

# ── Tab 1 – Scatter plots ─────────────────────────────────────────────────────
with tab1:
    pairs = [(feats[i], feats[j]) for i in range(len(feats)) for j in range(i + 1, len(feats))]
    if not pairs:
        st.info("Select at least 2 features to see scatter plots.")
    else:
        ncols = min(len(pairs), 2)
        nrows = (len(pairs) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                                 facecolor=plot_bg, squeeze=False)
        for idx, (fx, fy) in enumerate(pairs):
            ax = axes[idx // ncols][idx % ncols]
            for label, colour in PALETTE.items():
                sub = df_out[df_out["Prediction"] == label]
                ax.scatter(sub[fx], sub[fy], c=colour, label=label,
                           alpha=0.75, s=50, edgecolors="none")
            ax.set_xlabel(fx); ax.set_ylabel(fy)
            ax.set_title(f"{fx} vs {fy}")
            style_ax(ax)
        # Hide empty axes
        for idx in range(len(pairs), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
        fig.legend(handles=patches, loc="upper right", framealpha=0.3,
                   labelcolor=plot_fg, facecolor=plot_bg)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Tab 2 – Score distribution ────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_bg)
        for label, colour in PALETTE.items():
            sub = df_out[df_out["Prediction"] == label]["Anomaly Score"]
            ax.hist(sub, bins=20, color=colour, alpha=0.75, label=label, edgecolor="none")
        ax.axvline(0, color="#f9e2af", linewidth=1.2, linestyle="--", label="Decision boundary")
        ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
        ax.set_title("Score Distribution by Class")
        ax.legend(framealpha=0.3, labelcolor=plot_fg, facecolor=plot_bg)
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_bg)
        sorted_scores = df_out["Anomaly Score"].sort_values().values
        colours = [PALETTE["Anomaly"] if s < 0 else PALETTE["Normal"] for s in sorted_scores]
        ax.bar(range(len(sorted_scores)), sorted_scores, color=colours, width=1.0)
        ax.axhline(0, color="#f9e2af", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Customer Index (sorted)"); ax.set_ylabel("Anomaly Score")
        ax.set_title("Sorted Anomaly Scores")
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Tab 3 – Feature heatmap ───────────────────────────────────────────────────
with tab3:
    numeric_feats = df_out[feats + ["Anomaly Score"]].copy()
    numeric_feats["Is Anomaly"] = (df_out["Prediction"] == "Anomaly").astype(int)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_bg)
        corr = numeric_feats.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.4, linecolor=grid_col,
                    annot_kws={"size": 9, "color": plot_fg})
        ax.set_title("Feature Correlation Matrix", color=plot_fg)
        ax.tick_params(colors=plot_fg, labelsize=8)
        ax.figure.axes[-1].tick_params(colors=plot_fg)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_h2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_bg)
        group_means = df_out.groupby("Prediction")[feats].mean()
        sns.heatmap(group_means.T, annot=True, fmt=".1f", cmap="YlOrRd",
                    ax=ax, linewidths=0.4, linecolor=grid_col,
                    annot_kws={"size": 10, "color": "#1e1e2e"})
        ax.set_title("Mean Feature Values by Class", color=plot_fg)
        ax.tick_params(colors=plot_fg, labelsize=8)
        ax.figure.axes[-1].tick_params(colors=plot_fg)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Tab 4 – Detailed results ──────────────────────────────────────────────────
with tab4:
    filter_opt = st.radio("Show", ["All", "Anomalies Only", "Normal Only"],
                          horizontal=True)
    if filter_opt == "Anomalies Only":
        display_df = df_out[df_out["Prediction"] == "Anomaly"]
    elif filter_opt == "Normal Only":
        display_df = df_out[df_out["Prediction"] == "Normal"]
    else:
        display_df = df_out

    st.dataframe(
        display_df.sort_values("Anomaly Score").style
            .background_gradient(subset=["Anomaly Score"], cmap="RdYlGn")
            .applymap(lambda v: "color: #f38ba8; font-weight:600" if v == "Anomaly"
                      else "color: #a6e3a1", subset=["Prediction"]),
        use_container_width=True, height=400,
    )

    # Download
    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_bytes,
        file_name="anomaly_detection_results.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · scikit-learn Isolation Forest · Mall Customers Dataset")
