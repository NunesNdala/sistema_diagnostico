import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

NUM_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc",
]

CATEGORY_MAP = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
}

MODEL_FILES = {
    "Random Forest": "rf_kidney.pkl",
    "Regressao Logistica": "lr_kidney.pkl",
    "KNN": "knn_kidney.pkl",
}

# O Random Forest foi treinado com os dados em escala original (arvores nao
# precisam de normalizacao). Regressao Logistica e KNN foram treinados com
# os dados normalizados pelo StandardScaler (Z-score). Aplicar a normalizacao
# ao Random Forest quebra completamente as suas previsoes (faz o modelo
# prever quase sempre a classe maioritaria), por isso cada modelo recebe o
# conjunto de dados correspondente ao que foi usado no seu treino.
MODELS_REQUIRING_SCALING = {"Regressao Logistica", "KNN"}


st.set_page_config(
    page_title="Diagnostico CKD",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺",
)

# Paleta de cores com tema clinico/saude (verde-azulado, branco, tons suaves)
HEALTH_PRIMARY = "#0E3A53"      # azul-petroleo escuro
HEALTH_SECONDARY = "#1B6B93"    # azul-petroleo medio
HEALTH_ACCENT = "#E63946"       # vermelho clinico (alerta/CKD)
HEALTH_SUCCESS = "#2DBE82"      # verde saude (sem CKD)
HEALTH_BG = "#F2F6F8"           # fundo quase branco com toque azulado
HEALTH_CARD = "#FFFFFF"
HEALTH_TEXT = "#0D2B3E"

CHART_PALETTE = [HEALTH_PRIMARY, HEALTH_SECONDARY, "#5FA8C9", "#072536", "#A9CBDC"]

st.markdown(
    f"""
    <style>
        html, body, .stApp {{
            background-color: {HEALTH_BG} !important;
            color: {HEALTH_TEXT} !important;
            color-scheme: light !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {HEALTH_PRIMARY} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        h1, h2, h3 {{
            color: {HEALTH_PRIMARY} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
        }}
        /* Selectbox / dropdown (BaseWeb) - corrige contraste mesmo com dark mode do SO */
        [data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            color: {HEALTH_TEXT} !important;
            border: 1px solid {HEALTH_SECONDARY} !important;
        }}
        [data-baseweb="select"] * {{
            color: {HEALTH_TEXT} !important;
        }}
        [data-baseweb="popover"] {{
            background-color: #FFFFFF !important;
        }}
        ul[role="listbox"] {{
            background-color: #FFFFFF !important;
        }}
        ul[role="listbox"] li {{
            background-color: #FFFFFF !important;
            color: {HEALTH_TEXT} !important;
        }}
        ul[role="listbox"] li:hover {{
            background-color: {HEALTH_SECONDARY} !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stMetric"] {{
            background-color: {HEALTH_CARD} !important;
            border: 1px solid {HEALTH_SECONDARY};
            border-radius: 10px;
            padding: 10px;
        }}
        [data-testid="stMetric"] * {{
            color: {HEALTH_TEXT} !important;
        }}
        .stButton > button {{
            background-color: {HEALTH_PRIMARY};
            color: white !important;
            border-radius: 8px;
            border: none;
        }}
        .stButton > button:hover {{
            background-color: {HEALTH_SECONDARY};
            color: white !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {HEALTH_TEXT} !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: {HEALTH_PRIMARY} !important;
            font-weight: 700;
        }}
        /* Tabelas (st.dataframe) - garante texto escuro legivel sobre fundos claros */
        [data-testid="stDataFrame"] {{
            background-color: #FFFFFF !important;
        }}
        [data-testid="stDataFrame"] * {{
            color: {HEALTH_TEXT} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    scaler = joblib.load(BASE_DIR / "scaler_kidney.pkl")
    models = {}

    for name, filename in MODEL_FILES.items():
        path = BASE_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)

    if "Random Forest" not in models:
        raise FileNotFoundError("O modelo rf_kidney.pkl nao foi encontrado.")

    return scaler, models


def encode_features(df):
    encoded = df.copy()

    for col, mapping in CATEGORY_MAP.items():
        if col in encoded.columns:
            encoded[col] = encoded[col].astype(str).str.strip().map(mapping)

    encoded = encoded[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return encoded


def scale_numeric_features(df, scaler):
    scaled = df.copy()
    scaled[NUM_COLS] = scaler.transform(scaled[NUM_COLS])
    return scaled


@st.cache_data
def load_dataset():
    dataset_path = BASE_DIR / "kidney_clean.csv"
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    return df


def prepare_dataset(raw_df, scaler):
    df = raw_df.dropna(subset=["classification"]).copy()
    y = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"ckd": 1, "notckd": 0, "not ckd": 0})
    )

    valid = y.notna()
    x = encode_features(df.loc[valid, FEATURE_COLUMNS])
    y = y.loc[valid].astype(int)

    complete = x.notna().all(axis=1)
    x = x.loc[complete]
    y = y.loc[complete]

    x_scaled = scale_numeric_features(x, scaler)
    return x, x_scaled, y


def calculate_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None and len(set(y_true)) > 1 else np.nan

    baseline = max(float(y_true.mean()), 1 - float(y_true.mean()))
    standard_error = math.sqrt((baseline * (1 - baseline)) / len(y_true))
    z_score = (accuracy - baseline) / standard_error if standard_error else 0
    p_value = math.erfc(abs(z_score) / math.sqrt(2))

    return {
        "Acuracia": accuracy,
        "Precisao": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "Especificidade": specificity,
        "Z-Score": z_score,
        "P-Valor": p_value,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Matriz": cm,
    }


def evaluate_models(models, x_unscaled, x_scaled, y):
    rows = []
    matrices = {}

    for name, model in models.items():
        x = x_scaled if name in MODELS_REQUIRING_SCALING else x_unscaled
        y_pred = model.predict(x)
        y_proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = calculate_metrics(y, y_pred, y_proba)
        matrices[name] = metrics.pop("Matriz")
        rows.append({"Modelo": name, **metrics})

    return pd.DataFrame(rows).set_index("Modelo"), matrices


def patient_sidebar():
    st.sidebar.header("Dados do paciente")

    values = {
        "age": st.sidebar.slider("Idade", 2, 90, 50),
        "bp": st.sidebar.slider("Pressao arterial (bp)", 50, 180, 80),
        "sg": st.sidebar.selectbox("Gravidade especifica (sg)", [1.005, 1.010, 1.015, 1.020, 1.025]),
        "al": st.sidebar.slider("Albumina (al)", 0, 5, 0),
        "su": st.sidebar.slider("Acucar (su)", 0, 5, 0),
        "bgr": st.sidebar.slider("Glicose (bgr)", 22, 490, 120),
        "bu": st.sidebar.slider("Ureia (bu)", 1, 391, 40),
        "sc": st.sidebar.slider("Creatinina (sc)", 0.4, 76.0, 1.2),
        "sod": st.sidebar.slider("Sodio (sod)", 4, 163, 138),
        "pot": st.sidebar.slider("Potassio (pot)", 2.5, 7.6, 4.4),
        "hemo": st.sidebar.slider("Hemoglobina (hemo)", 3.1, 17.8, 13.0),
        "pcv": st.sidebar.slider("Volume globular (pcv)", 9, 54, 40),
        "wc": st.sidebar.slider("Globulos brancos (wc)", 2200, 26400, 8000),
        "rc": st.sidebar.slider("Globulos vermelhos (rc)", 2.1, 6.5, 4.8),
        "rbc": st.sidebar.selectbox("Globulos vermelhos (rbc)", ["normal", "abnormal"]),
        "pc": st.sidebar.selectbox("Celulas de pus (pc)", ["normal", "abnormal"]),
        "pcc": st.sidebar.selectbox("Aglomerados de pus (pcc)", ["notpresent", "present"]),
        "ba": st.sidebar.selectbox("Bacterias (ba)", ["notpresent", "present"]),
        "htn": st.sidebar.selectbox("Hipertensao (htn)", ["no", "yes"]),
        "dm": st.sidebar.selectbox("Diabetes (dm)", ["no", "yes"]),
        "cad": st.sidebar.selectbox("Doenca coronaria (cad)", ["no", "yes"]),
        "appet": st.sidebar.selectbox("Apetite (appet)", ["good", "poor"]),
        "pe": st.sidebar.selectbox("Edema (pe)", ["no", "yes"]),
        "ane": st.sidebar.selectbox("Anemia (ane)", ["no", "yes"]),
    }

    ordered = {col: values[col] for col in FEATURE_COLUMNS}
    patient_raw = pd.DataFrame([ordered])
    patient_encoded = encode_features(patient_raw)
    return values, patient_encoded


def show_prediction(patient_values, patient_encoded, scaler, model, model_name):
    if model_name in MODELS_REQUIRING_SCALING:
        patient_input = scale_numeric_features(patient_encoded, scaler)
    else:
        patient_input = patient_encoded

    prediction = model.predict(patient_input)[0]
    probability = model.predict_proba(patient_input)[0]

    st.subheader("Resultado")
    if prediction == 1:
        st.error("Doenca Renal Cronica (CKD) detectada")
        st.warning("Consulte um medico nefrologista para confirmacao e acompanhamento.")
    else:
        st.success("Sem Doenca Renal Cronica detectada")
        st.info("Mantenha acompanhamento medico regular e exames de rotina.")

    col1, col2 = st.columns(2)
    col1.metric("Probabilidade CKD", f"{probability[1] * 100:.1f}%")
    col2.metric("Probabilidade nao CKD", f"{probability[0] * 100:.1f}%")

    gauge_color = HEALTH_ACCENT if probability[1] >= 0.5 else HEALTH_SUCCESS
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[1] * 100,
        number={"suffix": "%", "font": {"color": HEALTH_TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": HEALTH_TEXT},
            "bar": {"color": gauge_color},
            "bgcolor": HEALTH_CARD,
            "steps": [
                {"range": [0, 50], "color": "#E4EEF3"},
                {"range": [50, 100], "color": "#FBE3E5"},
            ],
            "threshold": {
                "line": {"color": HEALTH_TEXT, "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
        title={"text": "Risco de CKD", "font": {"color": HEALTH_PRIMARY}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor=HEALTH_CARD,
        font_color=HEALTH_TEXT,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fatores de risco identificados")
    risks = []
    if patient_values["htn"] == "yes":
        risks.append("Hipertensao")
    if patient_values["dm"] == "yes":
        risks.append("Diabetes")
    if patient_values["ane"] == "yes":
        risks.append("Anemia")
    if patient_values["pe"] == "yes":
        risks.append("Edema")
    if patient_values["cad"] == "yes":
        risks.append("Doenca coronaria")
    if patient_values["rbc"] == "abnormal":
        risks.append("Globulos vermelhos anormais")
    if patient_values["pc"] == "abnormal":
        risks.append("Celulas de pus anormais")
    if patient_values["pcc"] == "present":
        risks.append("Aglomerados de pus presentes")
    if patient_values["ba"] == "present":
        risks.append("Bacterias presentes")
    if patient_values["appet"] == "poor":
        risks.append("Apetite reduzido")
    if patient_values["sc"] > 1.5:
        risks.append(f"Creatinina elevada ({patient_values['sc']})")
    if patient_values["hemo"] < 12.0:
        risks.append(f"Hemoglobina baixa ({patient_values['hemo']})")
    if patient_values["pcv"] < 36:
        risks.append(f"Volume globular baixo ({patient_values['pcv']})")
    if patient_values["rc"] < 4.5:
        risks.append(f"Globulos vermelhos (contagem) baixos ({patient_values['rc']})")
    if patient_values["sg"] < 1.015:
        risks.append(f"Gravidade especifica da urina baixa ({patient_values['sg']})")
    if patient_values["al"] > 0:
        risks.append(f"Albumina presente na urina ({patient_values['al']})")
    if patient_values["su"] > 0:
        risks.append(f"Acucar presente na urina ({patient_values['su']})")
    if patient_values["bgr"] > 140:
        risks.append(f"Glicemia elevada ({patient_values['bgr']})")
    if patient_values["bu"] > 50:
        risks.append(f"Ureia elevada ({patient_values['bu']})")
    if patient_values["sod"] < 135:
        risks.append(f"Sodio baixo ({patient_values['sod']})")
    if patient_values["pot"] > 5.5 or patient_values["pot"] < 3.5:
        risks.append(f"Potassio fora do intervalo normal ({patient_values['pot']})")
    if patient_values["bp"] > 90:
        risks.append(f"Pressao arterial elevada ({patient_values['bp']})")

    if risks:
        st.write(pd.DataFrame({"Fator": risks}))
    else:
        st.write("Nenhum fator de risco identificado.")


def show_model_comparison(results):
    display_cols = [
        "Acuracia", "Precisao", "Recall", "F1-Score", "AUC-ROC",
        "Especificidade", "Z-Score", "P-Valor", "TP", "TN", "FP", "FN",
    ]
    best_model = results["F1-Score"].idxmax()

    st.subheader("Comparacao entre modelos de treino")
    st.caption("Metricas calculadas com os modelos salvos e o ficheiro kidney_clean.csv.")
    st.success(f"Melhor modelo por F1-Score: {best_model}")

    st.dataframe(
        results[display_cols].style.set_properties(
            **{"color": HEALTH_TEXT, "background-color": "#FFFFFF"}
        ).format({
            "Acuracia": "{:.3f}",
            "Precisao": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-Score": "{:.3f}",
            "AUC-ROC": "{:.3f}",
            "Especificidade": "{:.3f}",
            "Z-Score": "{:.2f}",
            "P-Valor": "{:.4f}",
        }).highlight_max(
            subset=["Acuracia", "Precisao", "Recall", "F1-Score", "AUC-ROC", "Especificidade", "Z-Score"],
            color="#A9CBDC",
        ).highlight_min(
            subset=["P-Valor"], color="#A9CBDC"
        ),
        use_container_width=True,
    )

    chart_data = results[["Acuracia", "Precisao", "Recall", "F1-Score", "AUC-ROC"]].reset_index()
    chart_long = chart_data.melt(id_vars="Modelo", var_name="Metrica", value_name="Valor")

    fig = px.bar(
        chart_long,
        x="Metrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=CHART_PALETTE,
        title="Comparacao de metricas por modelo",
    )
    fig.update_layout(
        plot_bgcolor=HEALTH_CARD,
        paper_bgcolor=HEALTH_CARD,
        font_color=HEALTH_TEXT,
        yaxis_range=[0, 1.05],
        legend_title_text="Modelo",
        hovermode="x unified",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def show_confusion_matrices(matrices):
    st.subheader("Matrizes de confusao")
    cols = st.columns(min(3, len(matrices)))

    labels = ["nao CKD", "CKD"]

    for idx, (name, matrix) in enumerate(matrices.items()):
        with cols[idx % len(cols)]:
            st.markdown(f"**{name}**")

            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"Previsto: {lbl}" for lbl in labels],
                    y=[f"Real: {lbl}" for lbl in labels],
                    text=matrix,
                    texttemplate="%{text}",
                    textfont={"size": 18, "color": HEALTH_TEXT},
                    colorscale=[[0, "#FFFFFF"], [1, HEALTH_PRIMARY]],
                    showscale=False,
                    hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Casos: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor=HEALTH_CARD,
                plot_bgcolor=HEALTH_CARD,
                font_color=HEALTH_TEXT,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            tn, fp, fn, tp = matrix.ravel()
            st.caption(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")


def show_statistical_analysis(results, total_records):
    st.subheader("Analise estatistica: Z-Score")
    st.write(
        "O Z-Score compara a acuracia de cada modelo contra uma linha de base "
        "que escolhe sempre a classe maioritaria. Valores maiores indicam melhor "
        "distancia estatistica em relacao ao classificador base."
    )

    stat_df = results[["Acuracia", "Z-Score", "P-Valor", "TP", "TN", "FP", "FN"]].copy()
    stat_df["Significativo"] = np.where(
        stat_df["P-Valor"] < 0.05,
        "Sim (p < 0.05)",
        np.where(stat_df["P-Valor"] < 0.10, "Parcial (p < 0.10)", "Nao"),
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Registos avaliados", total_records)
    col2.metric("Maior Z-Score", f"{stat_df['Z-Score'].max():.2f}")
    col3.metric("Menor p-valor", f"{stat_df['P-Valor'].min():.4f}")

    st.dataframe(
        stat_df.style.set_properties(
            **{"color": HEALTH_TEXT, "background-color": "#FFFFFF"}
        ).format({
            "Acuracia": "{:.3f}",
            "Z-Score": "{:.2f}",
            "P-Valor": "{:.4f}",
        }).highlight_max(subset=["Z-Score"], color="#A9CBDC"),
        use_container_width=True,
    )
    fig = px.bar(
        results.reset_index(),
        x="Modelo",
        y="Z-Score",
        color="Modelo",
        text_auto=".2f",
        color_discrete_sequence=CHART_PALETTE,
        title="Z-Score por modelo (distancia em relacao a linha de base)",
    )
    fig.update_layout(
        plot_bgcolor=HEALTH_CARD,
        paper_bgcolor=HEALTH_CARD,
        font_color=HEALTH_TEXT,
        showlegend=False,
    )
    fig.add_hline(y=1.96, line_dash="dash", line_color=HEALTH_ACCENT,
                  annotation_text="Limiar p<0.05 (~1.96)", annotation_position="top right")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def main():
    scaler, models = load_assets()
    patient_values, patient_encoded = patient_sidebar()

    st.title("Sistema de Diagnostico de Doenca Renal Cronica")
    st.markdown("Introduza os dados do paciente e compare o desempenho dos modelos de treino.")

    raw_dataset = load_dataset()
    results = None
    matrices = None
    total_records = 0

    if raw_dataset is not None:
        x_unscaled_eval, x_scaled_eval, y_eval = prepare_dataset(raw_dataset, scaler)
        total_records = len(y_eval)
        if total_records > 0:
            results, matrices = evaluate_models(models, x_unscaled_eval, x_scaled_eval, y_eval)

    tab_diagnosis, tab_comparison, tab_matrix, tab_stats = st.tabs([
        "Diagnostico",
        "Comparacao de modelos",
        "Matriz de confusao",
        "Z-Score e metricas",
    ])

    with tab_diagnosis:
        selected_model_name = st.selectbox(
            "Modelo para previsao individual",
            list(models.keys()),
            index=list(models.keys()).index("Random Forest") if "Random Forest" in models else 0,
        )
        if st.button("Prever"):
            show_prediction(
                patient_values, patient_encoded, scaler,
                models[selected_model_name], selected_model_name,
            )

    if results is None or matrices is None:
        warning = (
            "Nao foi possivel calcular a comparacao. Confirme se kidney_clean.csv "
            "existe e contem as colunas esperadas."
        )
        with tab_comparison:
            st.warning(warning)
        with tab_matrix:
            st.warning(warning)
        with tab_stats:
            st.warning(warning)
        return

    with tab_comparison:
        show_model_comparison(results)

    with tab_matrix:
        show_confusion_matrices(matrices)

    with tab_stats:
        show_statistical_analysis(results, total_records)


if __name__ == "__main__":
    main()
