import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
import streamlit as st

PARQUET = "./data/ACRE_005_NP_8973-536.parquet"
SCALE = 0.001  # LAS default scale factor

st.set_page_config(page_title="LiDAR Explorer", layout="wide")
st.title("LiDAR Explorer — ACRE")

tab_3d, tab_tif = st.tabs(["Nuvem de pontos 3D", "GeoTIFF"])

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["x"] = df["X"] * SCALE
    df["y"] = df["Y"] * SCALE
    df["z"] = df["Z"] * SCALE
    return df


@st.cache_data
def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)


@st.cache_data
def load_tif(path: str):
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        band[band == nodata] = np.nan
    return band, bounds


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — Nuvem de pontos 3D
# ══════════════════════════════════════════════════════════════════════════════
with tab_3d:
    df_full = load_parquet(PARQUET)

    with st.sidebar:
        st.header("Visualização 3D")
        n_points   = st.slider("Pontos amostrados", 10_000, 200_000, 50_000, step=10_000)
        color_by   = st.selectbox("Colorir por", ["z", "intensity", "classification", "return_number"])
        colorscale = st.selectbox("Paleta de cores", ["Viridis", "Plasma", "Inferno", "RdYlGn", "Turbo"])
        point_size = st.slider("Tamanho do ponto", 1, 5, 2)

        st.divider()
        st.header("Filtros")

        z_min_all = float(df_full["z"].min())
        z_max_all = float(df_full["z"].max())
        z_range = st.slider("Elevação Z (m)", z_min_all, z_max_all, (z_min_all, z_max_all), step=0.5)

        int_min_all = int(df_full["intensity"].min())
        int_max_all = int(df_full["intensity"].max())
        int_range = st.slider("Intensity", int_min_all, int_max_all, (int_min_all, int_max_all))

        classes_all = sorted(df_full["classification"].unique().tolist())
        st.markdown("**Classificação**")
        classes_sel = [c for c in classes_all if st.checkbox(f"Classe {c}", value=True, key=f"cls_{c}")]

        returns_all = sorted(df_full["return_number"].unique().tolist())
        returns_sel = st.multiselect("Return number", options=returns_all, default=returns_all)

    df_sampled = sample(df_full, n_points)
    mask = (
        df_sampled["z"].between(*z_range)
        & df_sampled["intensity"].between(*int_range)
        & df_sampled["classification"].isin(classes_sel)
        & df_sampled["return_number"].isin(returns_sel)
    )
    df = df_sampled[mask]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pontos visíveis", f"{len(df):,}")
    c2.metric("Z mín (m)", f"{df['z'].min():.1f}" if len(df) else "—")
    c3.metric("Z máx (m)", f"{df['z'].max():.1f}" if len(df) else "—")
    c4.metric("Classes únicas", df["classification"].nunique() if len(df) else 0)

    if df.empty:
        st.warning("Nenhum ponto com os filtros aplicados.")
    else:
        fig = go.Figure(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="markers",
            marker=dict(size=point_size, color=df[color_by], colorscale=colorscale,
                        colorbar=dict(title=color_by), opacity=0.8),
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>",
        ))
        fig.update_layout(
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
            margin=dict(l=0, r=0, t=0, b=0), height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribuição de elevação (Z)")
            st.bar_chart(df["z"].value_counts(bins=60).sort_index())
        with col_b:
            st.subheader("Pontos por classe")
            st.bar_chart(df["classification"].value_counts().sort_index())


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — GeoTIFF
# ══════════════════════════════════════════════════════════════════════════════
with tab_tif:
    tif_files = sorted(glob.glob("./data/*.tif"))

    if not tif_files:
        st.info("Nenhum GeoTIFF encontrado em data/. Gere um com:\n\n"
                "`python -m src.console lidar parquet-to-tif data/<file>.parquet`")
    else:
        selected_tif = st.selectbox("Arquivo GeoTIFF", tif_files)
        tif_colorscale = st.selectbox("Paleta", ["Viridis", "Plasma", "Inferno", "RdYlGn", "Turbo"], key="tif_cs")

        band, bounds = load_tif(selected_tif)

        c1, c2, c3, c4 = st.columns(4)
        valid = band[~np.isnan(band)]
        c1.metric("Linhas × Colunas", f"{band.shape[0]} × {band.shape[1]}")
        c2.metric("Mín", f"{valid.min():.2f}" if len(valid) else "—")
        c3.metric("Máx", f"{valid.max():.2f}" if len(valid) else "—")
        c4.metric("Média", f"{valid.mean():.2f}" if len(valid) else "—")

        fig_tif = go.Figure(go.Heatmap(
            z=band,
            colorscale=tif_colorscale,
            colorbar=dict(title="valor"),
            hoverongaps=False,
        ))
        fig_tif.update_layout(
            xaxis=dict(title="coluna", scaleanchor="y"),
            yaxis=dict(title="linha", autorange="reversed"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
        )
        st.plotly_chart(fig_tif, use_container_width=True)
