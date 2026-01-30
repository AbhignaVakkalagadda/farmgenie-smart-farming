
"""
FarmGenie — Stable UI (headings set to white + balloons removed)

Robust fix for "Specifying the columns using strings is only supported for dataframes"
by inferring the exact column names expected by saved pipelines/column-transformers
and supplying a pandas.DataFrame with those column names to .transform()/.predict().
"""

import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime, timedelta
import base64
import traceback
import numpy as np
import keras
try:
    from tensorflow.keras.models import load_model
    _HAS_TF = True
except Exception:
    load_model = None
    _HAS_TF = False

# ---------------------------
# Config & paths
# ---------------------------
st.set_page_config(page_title="FarmGenie · Stable UI", layout="wide", initial_sidebar_state="expanded")
# Use same ROOT behaviour as your main logic
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
DB_PATH = os.path.join(ROOT, "shared_db.sqlite")

BG_PATH = r"C:\Users\vakka\Downloads\smart_farming_ai_project\smart_farming_ai_project\12.jpg"  # your path

# ---------------------------
# Helpers (logging / DB)
# ---------------------------
def path_expand(p):
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))

def safe_exists(p):
    try:
        return os.path.exists(p)
    except Exception:
        return False

def ensure_logs_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            agent TEXT,
            prediction TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(agent: str, prediction):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        timestamp = datetime.now().isoformat()
        cur.execute("INSERT INTO logs (timestamp, agent, prediction) VALUES (?, ?, ?)", (timestamp, agent, str(prediction)))
        conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"Logging failed: {e}")

def load_logs_df():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp","agent","prediction"])

ensure_logs_table()
df_logs = load_logs_df()

# ---------------------------
# Column / transformer introspection helpers
# ---------------------------

def _infer_columns_from_transformer(obj):
    """
    Try to infer expected feature names from a transformer/pipeline/columntransformer.
    Returns list of column names or None.
    """
    # 1) direct attribute
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass

    # 2) ColumnTransformer: inspect transformers_ tuples (if present)
    if hasattr(obj, "transformers_"):
        cols = []
        try:
            for name, trans, col_spec in obj.transformers_:
                # col_spec may be list/tuple/ndarray of names or indexes
                if isinstance(col_spec, (list, tuple, np.ndarray)):
                    for c in col_spec:
                        if isinstance(c, str):
                            cols.append(c)
        except Exception:
            cols = []
        if cols:
            # unique & preserve order
            seen = set(); out = []
            for c in cols:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out

    # 3) Pipeline: recurse into steps
    if hasattr(obj, "named_steps"):
        for _, step in obj.named_steps.items():
            cols = _infer_columns_from_transformer(step)
            if cols:
                return cols
    if hasattr(obj, "steps"):
        for _, step in getattr(obj, "steps"):
            cols = _infer_columns_from_transformer(step)
            if cols:
                return cols

    # 4) fallback: model may have n_features_in_
    if hasattr(obj, "n_features_in_"):
        try:
            n = int(obj.n_features_in_)
            return [f"col_{i}" for i in range(n)]
        except Exception:
            pass

    return None

def infer_expected_columns(model_or_transformer):
    """
    Try multiple locations (model pipeline, model.named_steps[0], model[0], scaler)
    to infer the expected input column names.
    """
    # Directly on object
    cols = _infer_columns_from_transformer(model_or_transformer)
    if cols:
        return cols

    # If it's a pipeline-like with a first transformer that expects columns
    if hasattr(model_or_transformer, "named_steps"):
        # prefer the first transformer step
        try:
            first = list(model_or_transformer.named_steps.values())[0]
            cols = _infer_columns_from_transformer(first)
            if cols:
                return cols
        except Exception:
            pass

    # indexes: maybe model_or_transformer[0] returns step
    try:
        first = model_or_transformer[0]
        cols = _infer_columns_from_transformer(first)
        if cols:
            return cols
    except Exception:
        pass

    return None

def _ensure_dataframe_with_cols(arr, cols):
    """
    Create a DataFrame from arr with column names 'cols', truncating or padding zeros as needed.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_need = len(cols)
    n_have = arr.shape[1]
    if n_have == n_need:
        return pd.DataFrame(arr, columns=cols)
    elif n_have > n_need:
        arr2 = arr[:, :n_need]
        return pd.DataFrame(arr2, columns=cols)
    else:
        pad = np.zeros((arr.shape[0], n_need - n_have))
        arr2 = np.concatenate([arr, pad], axis=1)
        return pd.DataFrame(arr2, columns=cols)

def prepare_input_df(expected_cols, input_values):
    """
    Build DataFrame for the expected_cols from a list/array input_values.
    Handles truncation/padding.
    """
    return _ensure_dataframe_with_cols(np.asarray(input_values), expected_cols)

def safe_model_predict(model, input_values):
    """
    Safely predict using 'model' which may be a Pipeline with ColumnTransformer expecting df columns.
    - model: sklearn/imblearn estimator or keras model
    - input_values: 1D list of feature values (in the order you intend)
    Returns prediction (model-dependent) or raises exception with diagnostics.
    """
    # if TF/Keras model (usual .predict on numpy), just call it later by the caller
    # But many of our sklearn pipelines need DataFrame with named columns
    cols = infer_expected_columns(model)
    if cols:
        df_in = prepare_input_df(cols, input_values)
        # try predict with dataframe
        return model.predict(df_in)
    else:
        # fallback: try numeric array, try reshape to (1, n)
        arr = np.asarray(input_values)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        try:
            return model.predict(arr)
        except Exception as e:
            # As a last resort, attempt to build a df with generic names if model has n_features_in_
            if hasattr(model, "n_features_in_"):
                n = int(model.n_features_in_)
                gen_cols = [f"c{i}" for i in range(n)]
                df_in2 = prepare_input_df(gen_cols, input_values)
                try:
                    return model.predict(df_in2)
                except Exception:
                    raise RuntimeError(f"safe_model_predict: final fallback failed; original error: {e}") from e
            raise

# ---------------------------
# CSS + background (same as your UI)
# ---------------------------
def inject_css():
    css = r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, Arial; color: #e9faff; }

    .white-heading { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; font-weight: 800 !important;
                     text-shadow: 0 1px 6px rgba(0,0,0,0.45); margin:0; padding:0; }

    .stSidebar { background: rgba(10,12,20,0.26) !important; backdrop-filter: blur(6px) !important; }
    .stSidebar .stButton, .stSidebar .stSlider { background: rgba(255,255,255,0.02) !important; }

    .glass { background: rgba(255,255,255,0.03) !important; border-radius:12px !important;
             padding:12px !important; border:1px solid rgba(255,255,255,0.04) !important;
             backdrop-filter: blur(6px) !important; color:#e9faff !important; }

    .result-card { background: rgba(255,255,255,0.30) !important; border-radius:12px !important; padding:12px !important;
                   color:#000 !important; font-weight:700; box-shadow: 0 6px 18px rgba(0,0,0,0.25) !important; margin-bottom:10px; }

    .stDataFrame, .stTable { color: #e9faff !important; background: rgba(0,0,0,0.34) !important; }
    .block-container { padding-top:12px !important; padding-left:18px !important; padding-right:18px !important; }
    .stApp > div:first-child { max-height:30px; overflow:visible; background:transparent !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

def set_bg_from_file(local_path, overlay_alpha=0.30):
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        ext = os.path.splitext(local_path)[1].lower().replace(".", "") or "jpeg"
        dataurl = f"data:image/{ext};base64,{b64}"
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,{overlay_alpha}), rgba(0,0,0,{overlay_alpha})), url("{dataurl}");
                background-size: cover; background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

if safe_exists(path_expand(BG_PATH)):
    set_bg_from_file(path_expand(BG_PATH), overlay_alpha=0.30)
else:
    st.sidebar.info("Background image not found at BG_PATH (silent fallback).")

# ---------------------------
# Sidebar: Inputs & actions (headings changed to white)
# ---------------------------
with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="white-heading">Manual Input for Prediction</h2>', unsafe_allow_html=True)

    temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)
    hum = st.slider("💧 Humidity (%)", 0.0, 100.0, 60.0)
    mois = st.slider("🌱 Moisture", 0, 1000, 500)
    ph = st.slider("⚗️ Soil pH", 0.0, 14.0, 6.5)
    rain = st.slider("☔ Rainfall (mm)", 0.0, 300.0, 100.0)

    st.markdown("---")
    st.markdown('<h3 class="white-heading">📦 Actions</h3>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Predict All Agents", use_container_width=True)

    st.markdown("---")
    df_logs_sidebar = load_logs_df()
    st.download_button(label="⬇️ Download Logs (CSV)", data=df_logs_sidebar.to_csv(index=False).encode("utf-8"), file_name="farmgenie_logs.csv", mime="text/csv")
    if st.button("🧹 Clear Logs"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM logs")
        conn.commit()
        conn.close()
        st.success("Logs cleared.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Main header (white)
# ---------------------------
st.markdown('<h1 class="white-heading">🌾 FarmGenie Multi-Agent Smart Farming Dashboard</h1>', unsafe_allow_html=True)
st.markdown("")  # spacing

# ---------------------------
# Prediction logic (keeps your original logic and filenames)
# ---------------------------
crop_labels = {
    0:"rice",1:"maize",2:"chickpea",3:"kidneybeans",4:"pigeonpeas",5:"mothbeans",
    6:"mungbean",7:"blackgram",8:"lentil",9:"pomegranate",10:"banana",11:"mango",
    12:"grapes",13:"watermelon",14:"muskmelon",15:"apple",16:"orange",17:"papaya",
    18:"coconut",19:"cotton",20:"jute",21:"coffee"
}

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_model_resource(path, is_dl=False):
    abs_path = path_expand(os.path.join(ROOT, path)) if not os.path.isabs(path) else path_expand(path)
    if not safe_exists(abs_path):
        return None
    try:
        if is_dl:
            if not _HAS_TF:
                return None
            return load_model(abs_path)
        else:
            return joblib.load(abs_path)
    except Exception:
        return None

# Run agents when button pressed
if run_btn:
    st.info("Running agents — this may take a few seconds depending on model sizes.")
    progress = st.progress(0)
    step = [0]
    total_steps = 6

    def step_update():
        step[0] += 1
        try:
            progress.progress(int(step[0] / total_steps * 100))
        except Exception:
            pass

    # 1) Crop recommendation
    try:
        step_update()
        model = load_model_resource("models/crop_model.pkl", is_dl=False)
        if model is not None:
            X = pd.DataFrame([[90, 42, 43, temp, hum, 6.5, rain]],
                             columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            y = model.predict(X)[0]
            crop_name = crop_labels.get(int(y), str(y))
            st.markdown(f"<div class='result-card'>🌿 Crop: {crop_name}</div>", unsafe_allow_html=True)
            log_prediction("crop_recommendation_agent", crop_name)
        else:
            st.warning("Crop model not available.")
    except Exception as e:
        st.error(f"Crop prediction error: {e}")

    # 2) Irrigation
    try:
        step_update()
        model = load_model_resource("models/irrigation_model.keras", is_dl=True)
        scaler = safe_load_joblib(os.path.join(ROOT, "models", "irrigation_scaler.pkl"))
        if model is not None and scaler is not None:
            try:
                # attempt robust scaling (same helper can be used here too)
                # but irrigation is Keras -> expects numpy; safe_scale returns numpy
                from sklearn.base import TransformerMixin  # local import ok
                # use prepare df if scaler expects columns else numpy
                cols = infer_expected_columns(scaler)
                if cols:
                    Xi = prepare_input_df(cols, [5.0, 2.0, 8.0])
                    scaled = scaler.transform(Xi)
                else:
                    scaled = np.asarray([[5.0, 2.0, 8.0]])
            except Exception:
                scaled = np.asarray([[5.0, 2.0, 8.0]])
            try:
                val = float(model.predict(scaled)[0][0])
                st.markdown(f"<div class='result-card'>💧 Irrigation: {round(val,2)} L</div>", unsafe_allow_html=True)
                log_prediction("irrigation_agent", round(val,2))
            except Exception as e:
                st.error(f"Irrigation predict error: {e}")
        else:
            st.warning("Irrigation model or scaler missing (or TensorFlow not installed).")
    except Exception as e:
        st.error(f"Irrigation error: {e}")

    # 3) Pest detection
    try:
        step_update()
        model = load_model_resource("models/pest_model.pkl", is_dl=False)
        scaler = safe_load_joblib(os.path.join(ROOT, "models", "pest_scaler.pkl"))
        if model is not None:
            try:
                # robust scaling
                if scaler is not None:
                    cols = infer_expected_columns(scaler)
                    if cols:
                        Xp = prepare_input_df(cols, [temp, 2450])
                        Xp_scaled = scaler.transform(Xp)
                    else:
                        Xp_scaled = scaler.transform(np.asarray([[temp, 2450]]))
                else:
                    Xp_scaled = np.asarray([[temp, 2450]])
            except Exception:
                Xp_scaled = np.asarray([[temp, 2450]])
            try:
                pred_val = model.predict(Xp_scaled)[0]
                try:
                    risk = "High" if bool(pred_val) else "Low"
                except Exception:
                    risk = str(pred_val)
                st.markdown(f"<div class='result-card'>🐞 Pest Risk: {risk}</div>", unsafe_allow_html=True)
                log_prediction("pest_detection_agent", risk)
            except Exception as e:
                st.error(f"Pest predict error: {e}")
        else:
            st.warning("Pest model or scaler missing.")
    except Exception as e:
        st.error(f"Pest detection error: {e}")

    # 4) Crop health  <-- FIXED: use safe_model_predict and DataFrame inference
    try:
        step_update()
        model = load_model_resource("models/crop_health_model.pkl", is_dl=False)
        scaler = safe_load_joblib(os.path.join(ROOT, "models", "crop_health_scaler.pkl"))
        if model is not None:
            # We need to pass the right shaped input to the pipeline. The pipeline might expect a dataframe
            # with specific column names. We'll attempt to infer those column names from the model/pipeline.
            try:
                # Build *raw* input vector (same order you used before)
                raw_input = [90, 42, 43, 6.5, hum, rain, temp]  # note ordering preserved

                # Attempt to infer columns from the *model pipeline* itself first
                expected_cols = infer_expected_columns(model)
                if expected_cols:
                    df_in = prepare_input_df(expected_cols, raw_input)
                    pred_arr = model.predict(df_in)
                else:
                    # If model didn't yield expected columns, maybe scaler expects them.
                    expected_cols_scaler = infer_expected_columns(scaler) if scaler is not None else None
                    if expected_cols_scaler:
                        df_in = prepare_input_df(expected_cols_scaler, raw_input)
                        # If the model is a pipeline with scaler inside, calling model.predict(df) will pass through scaler.
                        pred_arr = model.predict(df_in)
                    else:
                        # fallback: try safe_model_predict (it will attempt multiple fallbacks)
                        pred_arr = safe_model_predict(model, raw_input)

                # Interpret prediction robustly
                pred0 = pred_arr[0]
                if isinstance(pred0, (list, np.ndarray)):
                    pred0 = pred0[0]
                # If numeric (0/1), map to labels else use string
                try:
                    pred_int = int(np.rint(float(pred0)))
                    status = "Stressed" if pred_int == 1 else "Healthy"
                except Exception:
                    status = str(pred0)
                st.markdown(f"<div class='result-card'>🌡 Crop Health: {status}</div>", unsafe_allow_html=True)
                log_prediction("crop_health_agent", status)

            except Exception as e:
                # If we hit the old error, show helpful diagnostic
                msg = str(e)
                if "Specifying the columns using strings is only supported for dataframes" in msg or "only supported for dataframes" in msg:
                    st.error("Crop health prediction failed: transformer expects DataFrame with specific column names.")
                    st.error("Diagnostic: please inspect `models/crop_health_scaler.pkl` or the model pipeline in a notebook.")
                    st.write("Quick diagnostic steps (run in notebook):")
                    st.code("""
import joblib
sc = joblib.load('models/crop_health_scaler.pkl')
print(type(sc))
print(hasattr(sc,'feature_names_in_') and sc.feature_names_in_)
print(hasattr(sc,'transformers_') and sc.transformers_)
# For model pipeline:
m = joblib.load('models/crop_health_model.pkl')
print(type(m))
print(infer_expected_columns(m) or m.n_features_in_)
""")
                    st.text(traceback.format_exc())
                else:
                    st.error(f"Crop health predict error: {e}")
        else:
            st.warning("Crop health model or scaler missing.")
    except Exception as e:
        st.error(f"Crop health error: {e}")

    # 5) Recovery
    try:
        step_update()
        model = load_model_resource("models/recovery_model.keras", is_dl=True)
        scaler = safe_load_joblib(os.path.join(ROOT, "models", "recovery_scaler.pkl"))
        if model is not None:
            try:
                if scaler is not None:
                    cols = infer_expected_columns(scaler)
                    if cols:
                        Xr = prepare_input_df(cols, [hum, ph, rain, temp])
                        Xr_scaled = scaler.transform(Xr)
                    else:
                        Xr_scaled = scaler.transform(np.asarray([[hum, ph, rain, temp]]))
                else:
                    Xr_scaled = np.asarray([[hum, ph, rain, temp]])
            except Exception:
                Xr_scaled = np.asarray([[hum, ph, rain, temp]])
            try:
                days_raw = model.predict(Xr_scaled)[0]
                if isinstance(days_raw, (list, np.ndarray)):
                    days_raw = days_raw[0]
                days = int(np.rint(float(days_raw)))
                st.markdown(f"<div class='result-card'>⏳ Recovery: {days} days</div>", unsafe_allow_html=True)
                log_prediction("recovery_agent", f"{days} days")
            except Exception as e:
                st.error(f"Recovery predict error: {e}")
        else:
            st.warning("Recovery model or scaler missing (or TensorFlow not installed).")
    except Exception as e:
        st.error(f"Recovery error: {e}")

    # 6) Moisture forecast
    try:
        step_update()
        model = load_model_resource("models/moisture_lstm_daily.keras", is_dl=True)
        scaler = safe_load_joblib(os.path.join(ROOT, "models", "moisture_scaler_daily.pkl"))
        if model is not None:
            try:
                if scaler is not None:
                    cols = infer_expected_columns(scaler)
                    if cols:
                        Xm = prepare_input_df(cols, [mois])
                        Xm_scaled = scaler.transform(Xm)
                    else:
                        Xm_scaled = scaler.transform(np.asarray([[mois]]))
                else:
                    Xm_scaled = np.asarray([[mois]])
            except Exception:
                Xm_scaled = np.asarray([[mois]])
            try:
                moist = float(model.predict(Xm_scaled)[0][0])
                st.markdown(f"<div class='result-card'>💦 Next-Day Moisture: {round(moist,2)}</div>", unsafe_allow_html=True)
                log_prediction("moisture_forecast_agent", round(moist,2))
            except Exception as e:
                st.error(f"Moisture predict error: {e}")
        else:
            st.warning("Moisture model or scaler missing (or TensorFlow not installed).")
    except Exception as e:
        st.error(f"Moisture forecast error: {e}")

    progress.progress(100)
    # Balloons removed
    df_logs = load_logs_df()

# ---------------------------
# Logs & Trends (white heading)
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<h2 class="white-heading">📋 Logs & Trends</h2>', unsafe_allow_html=True)

logs_col, trends_col = st.columns([0.6, 0.4])

with logs_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    df_logs = load_logs_df()
    if df_logs.empty:
        st.info("No logs yet. Run predictions to populate logs.")
    else:
        st.dataframe(df_logs.head(200), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with trends_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("Select agent & date range to view numeric trends")
    df_logs = load_logs_df()
    if df_logs.empty:
        st.info("No trend data.")
    else:
        agents = df_logs["agent"].unique().tolist()
        sel_agent = st.selectbox("Agent", agents)
        max_date = pd.to_datetime(df_logs["timestamp"]).max()
        min_date = pd.to_datetime(df_logs["timestamp"]).min()
        d0 = st.date_input("From", value=(min_date.date() if not pd.isna(min_date) else datetime.now().date() - timedelta(days=30)))
        d1 = st.date_input("To", value=(max_date.date() if not pd.isna(max_date) else datetime.now().date()))
        smoothing = st.slider("Smoothing window (points)", 1, 21, 1)

        if sel_agent:
            df_a = df_logs[df_logs["agent"]==sel_agent].copy()
            df_a["ts"] = pd.to_datetime(df_a["timestamp"])
            df_a = df_a.set_index("ts").sort_index()
            df_a = df_a[(df_a.index.date >= d0) & (df_a.index.date <= d1)]
            df_a["numeric"] = pd.to_numeric(df_a["prediction"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
            df_plot = df_a.dropna(subset=["numeric"])
            if not df_plot.empty:
                if smoothing > 1:
                    df_plot["numeric_smooth"] = df_plot["numeric"].rolling(smoothing, min_periods=1).mean()
                    st.line_chart(df_plot["numeric_smooth"])
                else:
                    st.line_chart(df_plot["numeric"])
            else:
                st.info("No numeric data for selected agent/date range.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("FarmGenie")