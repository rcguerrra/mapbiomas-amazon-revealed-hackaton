import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import colors as pcolors
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from config import load_gcp_service_account_dict
from lidar import laz_to_parquet
from utils.storage_client import StorageClient

DEFAULT_LIDAR_LAZ_URI = (
    "gs://amazon-revealed/Point-Cloud/01_ENTREGA_23_08_2023/NP/ACRE_005_NP_8976-536.laz"
)

POINT_CLOUD_LIST_PREFIX_GS = "gs://amazon-revealed/Point-Cloud"
POINT_CLOUD_MAX_LAZ_BYTES = 50 * 1024 * 1024
POINT_CLOUD_LIST_LIMIT = 100

load_dotenv()


def _sync_streamlit_secrets_to_environ() -> None:
    """Expose Streamlit Cloud / secrets.toml keys as env vars for credential loaders."""
    try:
        if hasattr(st, "secrets"):
            for key in (
                "GCP_SERVICE_ACCOUNT_JSON",
                "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "MBENGINE_GCP_SERVICE_ACCOUNT",
            ):
                if key in st.secrets:
                    value = st.secrets[key]
                    if isinstance(value, dict):
                        os.environ[key] = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, str):
                        os.environ[key] = value
                    else:
                        os.environ[key] = json.dumps(value, ensure_ascii=False)
    except Exception:
        pass


_sync_streamlit_secrets_to_environ()

SCALE = 0.001  # LAS default scale factor

st.set_page_config(page_title="LiDAR Explorer", layout="wide")
st.title("LiDAR Explorer")

tab_3d, tab_map = st.tabs(["Nuvem de pontos 3D", "MapLibre"])

# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_storage_path(path: str) -> str:
    if path.startswith("gcs://"):
        return "gs://" + path[len("gcs://") :]
    return path


def _build_storage_client() -> StorageClient:
    return StorageClient(gcp_credentials=load_gcp_service_account_dict())


def ensure_local_parquet_from_remote_laz(
    remote_uri: str,
    *,
    force_download: bool = False,
) -> Path:
    """
    Download LAZ/LAS from gs:// or s3:// into data/ when needed, convert to Parquet,
    and return the local Parquet path.
    """
    remote = _normalize_storage_path(remote_uri.strip())
    if not remote.startswith(("gs://", "s3://")):
        raise ValueError("LAZ URI must start with gs:// or s3://.")

    client = _build_storage_client()
    if not client.exists(remote):
        raise FileNotFoundError(f"Remote file not found: {remote}")

    filename = remote.rstrip("/").rsplit("/", 1)[-1]
    if not filename:
        raise ValueError("Could not infer filename from URI.")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    local_laz = data_dir / filename
    local_parquet = data_dir / f"{Path(filename).stem}.parquet"

    if force_download or not local_laz.is_file():
        local_laz.write_bytes(client.read(remote))

    need_convert = not local_parquet.is_file()
    if local_laz.is_file() and local_parquet.is_file():
        if local_laz.stat().st_mtime > local_parquet.stat().st_mtime:
            need_convert = True

    if need_convert:
        laz_to_parquet(str(local_laz), local_parquet)

    load_parquet.clear()
    return local_parquet


@st.cache_data(ttl=600, show_spinner="Listing Point-Cloud objects…")
def list_point_cloud_laz_catalog() -> list[tuple[str, int]]:
    client = _build_storage_client()
    return client.list_gcs_files_filtered(
        POINT_CLOUD_LIST_PREFIX_GS,
        min_size_bytes=None,
        max_size_bytes=POINT_CLOUD_MAX_LAZ_BYTES,
        suffixes=(".laz", ".las"),
        sort_descending=True,
        limit=POINT_CLOUD_LIST_LIMIT,
    )


def _format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _df_from_remote_laz_pipeline(
    laz_uri: str,
    *,
    load_laz_clicked: bool,
    force_laz_sync: bool,
) -> tuple[pd.DataFrame, str]:
    if not laz_uri.strip():
        st.info("Selecione ou informe a URI do arquivo .laz na storage (gs:// ou s3://).")
        st.stop()

    uri_norm = _normalize_storage_path(laz_uri.strip())
    loaded_uri = st.session_state.get("laz_loaded_uri_norm")
    parquet_cached = st.session_state.get("laz_storage_parquet_path")

    if loaded_uri != uri_norm and not load_laz_clicked:
        st.info(
            "Clique em **Carregar arquivo** para baixar da storage e converter para Parquet."
        )
        st.stop()

    reuse_local_parquet = (
        loaded_uri == uri_norm
        and parquet_cached
        and Path(parquet_cached).is_file()
        and not force_laz_sync
        and not load_laz_clicked
    )

    if reuse_local_parquet:
        parquet_path = Path(parquet_cached)
    else:
        try:
            with st.spinner(
                "Checking storage, downloading LAZ if needed, converting to Parquet…"
            ):
                parquet_path = ensure_local_parquet_from_remote_laz(
                    laz_uri.strip(), force_download=force_laz_sync
                )
        except Exception as exc:
            st.error(f"Erro ao sincronizar LAZ / Parquet: {exc}")
            st.stop()
        st.session_state["laz_loaded_uri_norm"] = uri_norm
        st.session_state["laz_storage_parquet_path"] = str(parquet_path)

    try:
        df_full = load_parquet(str(parquet_path))
    except Exception as exc:
        st.error(f"Erro ao carregar Parquet: {exc}")
        st.stop()
    source_label = f"{uri_norm} → {parquet_path}"
    return df_full, source_label


def _standardize_point_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    if "x" in cols and "y" in cols and "z" in cols:
        x_raw = df[cols["x"]].astype(float)
        y_raw = df[cols["y"]].astype(float)
        z_raw = df[cols["z"]].astype(float)
        if np.nanmax(np.abs(x_raw.to_numpy())) > 10000:
            df["x"] = x_raw * SCALE
            df["y"] = y_raw * SCALE
            df["z"] = z_raw * SCALE
        else:
            df["x"] = x_raw
            df["y"] = y_raw
            df["z"] = z_raw
    elif "longitude" in cols and "latitude" in cols and "elevacao_z" in cols:
        df["x"] = df[cols["longitude"]].astype(float)
        df["y"] = df[cols["latitude"]].astype(float)
        df["z"] = df[cols["elevacao_z"]].astype(float)
    else:
        raise ValueError(
            "Dataset must contain one of: (X,Y,Z), (x,y,z), or "
            "(longitude,latitude,elevacao_z)."
        )

    if "intensity" not in cols:
        if "intensity" not in df.columns:
            df["intensity"] = 0
    else:
        df["intensity"] = df[cols["intensity"]]

    if "classification" not in cols:
        if "classification" not in df.columns:
            df["classification"] = 0
    else:
        df["classification"] = df[cols["classification"]]

    if "return_number" not in cols:
        if "return_number" not in df.columns:
            df["return_number"] = 1
    else:
        df["return_number"] = df[cols["return_number"]]

    return df


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    normalized_path = _normalize_storage_path(path)
    df = pd.read_parquet(normalized_path)
    return _standardize_point_columns(df)


@st.cache_data
def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)


@st.cache_data
def prepare_map_points(
    df: pd.DataFrame, color_by: str, colorscale: str, z_exaggeration: float
) -> tuple[list[dict], float, float]:
    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()
    z = df["z"].astype(float).to_numpy()

    # If coordinates are already geographic, use them directly.
    is_geographic = (
        np.nanmin(x) >= -180 and np.nanmax(x) <= 180 and np.nanmin(y) >= -90 and np.nanmax(y) <= 90
    )

    if is_geographic:
        lon = x
        lat = y
    else:
        # Fallback: treat XY as local metric coordinates and anchor in Acre.
        lon0 = -67.8
        lat0 = -9.8
        dx = x - np.nanmean(x)
        dy = y - np.nanmean(y)
        lon = lon0 + (dx / (111320.0 * np.cos(np.radians(lat0))))
        lat = lat0 + (dy / 110540.0)

    zmin = float(np.nanmin(z))

    c = df[color_by].astype(float).to_numpy()
    cmin = float(np.nanmin(c))
    cmax = float(np.nanmax(c))
    cspan = max(cmax - cmin, 1e-9)
    cn = (c - cmin) / cspan

    sampled_colors = pcolors.sample_colorscale(colorscale, cn.tolist())

    def _rgba_from_plotly(color: str) -> list[int]:
        match = re.findall(r"[\d\.]+", color)
        r, g, b = [int(float(v)) for v in match[:3]]
        return [r, g, b, 180]

    points = [
        {
            "position": [float(lon_i), float(lat_i), float((z_i - zmin) * z_exaggeration)],
            "color": _rgba_from_plotly(color_i),
        }
        for lon_i, lat_i, z_i, color_i in zip(lon, lat, z, sampled_colors)
    ]
    return points, float(np.nanmean(lon)), float(np.nanmean(lat))


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — Nuvem de pontos 3D
# ══════════════════════════════════════════════════════════════════════════════
with tab_3d:
    with st.sidebar:
        st.header("Visualização 3D")
        data_source = st.selectbox(
            "Fonte",
            [
                "Point-Cloud em gs://amazon-revealed (até 50 MB)",
                "LAZ na Storage (URI manual)",
            ],
        )
        n_points = st.slider("Pontos amostrados", 10_000, 200_000, 50_000, step=10_000)

        if data_source == "Point-Cloud em gs://amazon-revealed (até 50 MB)":
            try:
                entries = list_point_cloud_laz_catalog()
            except Exception as exc:
                st.error(f"Erro ao listar {POINT_CLOUD_LIST_PREFIX_GS}: {exc}")
                st.stop()
            if not entries:
                st.warning(
                    f"Nenhum arquivo .laz/.las até 50 MB em {POINT_CLOUD_LIST_PREFIX_GS}/ "
                    "(listagem recursiva)."
                )
                st.stop()
            by_uri = dict(entries)
            laz_uri = st.selectbox(
                "Arquivo LAZ",
                options=list(by_uri.keys()),
                format_func=lambda u: f"{u.rsplit('/', 1)[-1]} — {_format_size_mb(by_uri[u])}",
                help=(
                    f"Até {POINT_CLOUD_LIST_LIMIT} arquivos de até 50 MB, "
                    "do maior para o menor."
                ),
            )
            force_laz_sync = st.checkbox(
                "Force re-download and re-convert",
                value=False,
                help="Ignore cached local LAZ/Parquet for this URI and sync again from storage.",
            )
            load_laz_clicked = st.button(
                "Carregar arquivo",
                type="primary",
                use_container_width=True,
                key="load_laz_point_cloud",
                help="Download from storage (if needed) and convert to Parquet before viewing.",
            )
            df_full, source_label = _df_from_remote_laz_pipeline(
                laz_uri,
                load_laz_clicked=load_laz_clicked,
                force_laz_sync=force_laz_sync,
            )
        elif data_source == "LAZ na Storage (URI manual)":
            default_laz_uri = os.getenv("LIDAR_LAZ_URI", DEFAULT_LIDAR_LAZ_URI)
            laz_uri = st.text_input(
                "URI do LAZ",
                value=default_laz_uri,
                placeholder="gs://bucket/path/arquivo.laz",
                help="The object must exist on storage. It is saved under data/ and converted to Parquet for the viewer.",
            ).strip()
            force_laz_sync = st.checkbox(
                "Force re-download and re-convert",
                value=False,
                help="Ignore cached local LAZ/Parquet for this URI and sync again from storage.",
            )
            load_laz_clicked = st.button(
                "Carregar arquivo",
                type="primary",
                use_container_width=True,
                key="load_laz_manual_uri",
                help="Download from storage (if needed) and convert to Parquet before viewing.",
            )
            df_full, source_label = _df_from_remote_laz_pipeline(
                laz_uri,
                load_laz_clicked=load_laz_clicked,
                force_laz_sync=force_laz_sync,
            )

        if df_full.empty:
            st.warning("A fonte selecionada não retornou dados.")
            st.stop()

        st.caption(f"Fonte ativa: {source_label}")
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
# ABA 3 — MapLibre
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Mapa base (Google Satélite) + pontos 3D")
    z_exaggeration = st.slider("Exagero vertical", 0.2, 8.0, 2.0, step=0.2, key="map_z_ex")
    st.caption(f"Usando {n_points:,} pontos (controle 'Pontos amostrados' da barra lateral).")

    if df.empty:
        st.warning("Nenhum ponto para desenhar no mapa com os filtros atuais.")
        st.stop()

    points, center_lon, center_lat = prepare_map_points(df, color_by, colorscale, z_exaggeration)
    points_json = json.dumps(points)

    map_html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link
        href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"
        rel="stylesheet"
      />
      <style>
        html, body, #map { margin: 0; height: 100%; width: 100%; }
      </style>
    </head>
    <body>
      <div id="map"></div>
      <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
      <script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
      <script src="https://unpkg.com/@deck.gl/mapbox@8.9.36/dist.min.js"></script>
      <script>
        const map = new maplibregl.Map({
          container: 'map',
          style: {
            version: 8,
            sources: {
              'google-sat': {
                type: 'raster',
                tiles: ['https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'],
                tileSize: 256,
                attribution: 'Google Satellite'
              }
            },
            layers: [
              { id: 'google-sat-layer', type: 'raster', source: 'google-sat' }
            ]
          },
          center: [__CENTER_LON__, __CENTER_LAT__],
          zoom: 12,
          pitch: 55,
          bearing: -15
        });

        const pointCloudLayer = new deck.PointCloudLayer({
          id: 'lidar-point-cloud',
          data: __POINTS_JSON__,
          getPosition: d => d.position,
          getColor: d => d.color,
          pointSize: 0.8,
          coordinateSystem: deck.COORDINATE_SYSTEM.LNGLAT,
          pickable: false
        });

        map.on('load', () => {
          const overlay = new deck.MapboxOverlay({
            interleaved: true,
            layers: [pointCloudLayer]
          });
          map.addControl(overlay);
        });
      </script>
    </body>
    </html>
    """
    map_html = (
        map_html
        .replace("__POINTS_JSON__", points_json)
        .replace("__CENTER_LON__", str(center_lon))
        .replace("__CENTER_LAT__", str(center_lat))
    )
    components.html(map_html, height=650)
