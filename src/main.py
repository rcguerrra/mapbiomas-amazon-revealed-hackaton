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
    "gs://amazon-revealed/Point-Cloud/32_ENTREGA_17-12-2024/NP/MARAJO_009_NP_9892-684.laz"
)

POINT_CLOUD_COVER_LAZ_URI = DEFAULT_LIDAR_LAZ_URI

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


@st.cache_data(ttl=600, show_spinner=False)
def _gcs_file_size_bytes(gs_uri: str) -> int:
    client = _build_storage_client()
    return client.gcs_object_size_bytes(_normalize_storage_path(gs_uri))


def _catalog_with_cover_first(
    entries: list[tuple[str, int]],
    cover_uri: str,
) -> list[tuple[str, int]]:
    """
    Put the cover URI first (default selection). De-duplicate; trim to POINT_CLOUD_LIST_LIMIT.
    If the cover is not in the catalog list (e.g. size filter), size is fetched via GCS metadata.
    """
    cover_norm = _normalize_storage_path(cover_uri.strip())
    rest = [
        (u, s)
        for u, s in entries
        if _normalize_storage_path(u) != cover_norm
    ]
    size_cover: int | None = None
    for u, s in entries:
        if _normalize_storage_path(u) == cover_norm:
            size_cover = s
            break
    if size_cover is None:
        try:
            size_cover = _gcs_file_size_bytes(cover_norm)
        except Exception:
            size_cover = 0
    ordered = [(cover_norm, size_cover)] + rest
    return ordered[:POINT_CLOUD_LIST_LIMIT]


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
    """
    Normalize coordinate/attribute columns and return a minimal DataFrame
    with only the 6 columns needed for visualization, all in compact dtypes.
    Avoids float64 and drops all unreferenced columns immediately.
    """
    cols = {c.lower(): c for c in df.columns}
    n = len(df)

    if "x" in cols and "y" in cols and "z" in cols:
        x_raw = df[cols["x"]].to_numpy(dtype=np.float64)
        y_raw = df[cols["y"]].to_numpy(dtype=np.float64)
        z_raw = df[cols["z"]].to_numpy(dtype=np.float64)
        if np.nanmax(np.abs(x_raw)) > 10000:
            x = (x_raw * SCALE).astype(np.float32)
            y = (y_raw * SCALE).astype(np.float32)
            z = (z_raw * SCALE).astype(np.float32)
        else:
            x = x_raw.astype(np.float32)
            y = y_raw.astype(np.float32)
            z = z_raw.astype(np.float32)
        del x_raw, y_raw, z_raw
    elif "longitude" in cols and "latitude" in cols and "elevacao_z" in cols:
        x = df[cols["longitude"]].to_numpy(dtype=np.float32)
        y = df[cols["latitude"]].to_numpy(dtype=np.float32)
        z = df[cols["elevacao_z"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError(
            "Dataset must contain one of: (X,Y,Z), (x,y,z), or "
            "(longitude,latitude,elevacao_z)."
        )

    intensity = (
        df[cols["intensity"]].to_numpy(dtype=np.uint16)
        if "intensity" in cols
        else np.zeros(n, dtype=np.uint16)
    )
    classification = (
        df[cols["classification"]].to_numpy(dtype=np.uint8)
        if "classification" in cols
        else np.zeros(n, dtype=np.uint8)
    )
    return_number = (
        df[cols["return_number"]].to_numpy(dtype=np.uint8)
        if "return_number" in cols
        else np.ones(n, dtype=np.uint8)
    )

    return pd.DataFrame(
        {"x": x, "y": y, "z": z,
         "intensity": intensity,
         "classification": classification,
         "return_number": return_number},
    )


def _select_parquet_columns(path: str) -> list[str] | None:
    """Return only the columns needed for visualization; None means read all."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        available = {c.lower(): c for c in schema.names}
        needed: list[str] = []
        # coordinate candidates
        for name in ("x", "y", "z", "longitude", "latitude", "elevacao_z"):
            if name in available:
                needed.append(available[name])
        for name in ("intensity", "classification", "return_number"):
            if name in available:
                needed.append(available[name])
        return needed if needed else None
    except Exception:
        return None


@st.cache_data(max_entries=1)
def load_parquet(path: str) -> pd.DataFrame:
    normalized_path = _normalize_storage_path(path)
    cols = _select_parquet_columns(normalized_path)
    df = pd.read_parquet(normalized_path, columns=cols)
    return _standardize_point_columns(df)


@st.cache_data(max_entries=2)
def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)


@st.cache_data(max_entries=1)
def prepare_map_points(
    df: pd.DataFrame, color_by: str, colorscale: str, z_exaggeration: float
) -> tuple[str, float, float]:
    """
    Build a deck.gl-ready JSON string for the PointCloudLayer.
    Returns (points_json, center_lon, center_lat).
    Uses numpy throughout to avoid per-point Python objects.
    """
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    z = df["z"].to_numpy(dtype=np.float32)

    is_geographic = (
        float(np.nanmin(x)) >= -180 and float(np.nanmax(x)) <= 180
        and float(np.nanmin(y)) >= -90 and float(np.nanmax(y)) <= 90
    )

    if is_geographic:
        lon = x
        lat = y
    else:
        lon0 = -67.8
        lat0 = -9.8
        dx = x - np.nanmean(x)
        dy = y - np.nanmean(y)
        lon = (lon0 + dx / (111320.0 * np.cos(np.radians(lat0)))).astype(np.float32)
        lat = (lat0 + dy / 110540.0).astype(np.float32)
        del dx, dy

    zmin = float(np.nanmin(z))
    z_shifted = ((z - zmin) * z_exaggeration).astype(np.float32)
    del z

    c = df[color_by].to_numpy(dtype=np.float32)
    cmin = float(np.nanmin(c))
    cspan = max(float(np.nanmax(c)) - cmin, 1e-9)
    cn = ((c - cmin) / cspan).tolist()
    del c

    sampled_colors = pcolors.sample_colorscale(colorscale, cn)
    del cn

    # Parse RGB values into a uint8 numpy array — avoids per-point Python dicts.
    pattern = re.compile(r"[\d\.]+")
    rgb = np.array(
        [[int(float(v)) for v in pattern.findall(col)[:3]] for col in sampled_colors],
        dtype=np.uint8,
    )
    del sampled_colors

    # Build JSON string with vectorised string formatting.
    rows = [
        f'{{"position":[{lo:.6f},{la:.6f},{zs:.2f}],"color":[{r},{g},{b},180]}}'
        for lo, la, zs, (r, g, b) in zip(lon, lat, z_shifted, rgb)
    ]
    points_json = "[" + ",".join(rows) + "]"
    del rows, rgb, z_shifted

    return points_json, float(np.nanmean(lon)), float(np.nanmean(lat))


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
        n_points = st.slider("Pontos amostrados", 5_000, 100_000, 30_000, step=5_000)

        if data_source == "Point-Cloud em gs://amazon-revealed (até 50 MB)":
            try:
                entries = list_point_cloud_laz_catalog()
            except Exception as exc:
                st.error(f"Erro ao listar {POINT_CLOUD_LIST_PREFIX_GS}: {exc}")
                st.stop()
            entries = _catalog_with_cover_first(entries, POINT_CLOUD_COVER_LAZ_URI)
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
                index=0,
                format_func=lambda u: (
                    f"★ {u.rsplit('/', 1)[-1]} — {_format_size_mb(by_uri[u])}"
                    if _normalize_storage_path(u) == _normalize_storage_path(POINT_CLOUD_COVER_LAZ_URI)
                    else f"{u.rsplit('/', 1)[-1]} — {_format_size_mb(by_uri[u])}"
                ),
                help=(
                    f"Primeiro item: capa (padrão). Até {POINT_CLOUD_LIST_LIMIT} arquivos "
                    "de até 50 MB (exceto a capa, sempre listada), do maior para o menor."
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
        colorscale = st.selectbox(
            "Paleta de cores",
            ["Viridis", "Plasma", "Inferno", "RdYlGn", "Turbo"],
            index=2,
            help="Default: Inferno.",
        )
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

    points_json, center_lon, center_lat = prepare_map_points(df, color_by, colorscale, z_exaggeration)

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
