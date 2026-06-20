import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
from scipy import stats

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Diagnóstico CKD',
    page_icon='🫀',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
        border-bottom: 3px solid #2196F3;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1a3a5c 0%, #2d6a9f 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
    }
    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #2196F3;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .best-model-badge {
        background: #e8f5e9;
        border: 2px solid #4caf50;
        color: #2e7d32;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tab-subheader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3a5c;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load principal model & scaler ────────────────────────────────────────────
@st.cache_resource
def load_main_model():
    rf = joblib.load('rf_kidney.pkl')
    scaler = joblib.load('scaler_kidney.pkl')
    return rf, scaler

# ─── Generate synthetic benchmark data (mirrors CKD dataset statistics) ───────
@st.cache_data
def generate_benchmark_data():
    """
    Cria dados sintéticos para benchmarking dos modelos quando o dataset
    real não está disponível. Usa as distribuições conhecidas do UCI CKD dataset.
    """
    np.random.seed(42)
    n = 400
    # 60% CKD, 40% não-CKD (proporção típica do dataset UCI)
    y = np.array([1]*240 + [0]*160)

    X_ckd = np.column_stack([
        np.random.normal(55, 15, 240),   # age
        np.random.normal(90, 20, 240),   # bp (elevada)
        np.random.choice([1.005,1.010,1.015,1.020,1.025],240, p=[0.4,0.3,0.2,0.07,0.03]),
        np.random.choice([0,1,2,3,4,5], 240, p=[0.1,0.2,0.3,0.2,0.12,0.08]),
        np.random.choice([0,1,2,3,4,5], 240, p=[0.2,0.25,0.25,0.15,0.1,0.05]),
        np.random.normal(150, 60, 240),  # bgr
        np.random.normal(90, 50, 240),   # bu
        np.random.normal(5.0, 4.0, 240), # sc
        np.random.normal(130, 15, 240),  # sod
        np.random.normal(4.8, 1.0, 240), # pot
        np.random.normal(10.0, 2.5, 240),# hemo
        np.random.normal(30, 8, 240),    # pcv
        np.random.normal(9000, 4000,240),# wc
        np.random.normal(3.8, 0.8, 240), # rc
        np.random.binomial(1, 0.7, 240), # rbc abnormal
        np.random.binomial(1, 0.65,240), # pc abnormal
        np.random.binomial(1, 0.45,240), # pcc present
        np.random.binomial(1, 0.40,240), # ba present
        np.random.binomial(1, 0.75,240), # htn
        np.random.binomial(1, 0.65,240), # dm
        np.random.binomial(1, 0.30,240), # cad
        np.random.binomial(1, 0.55,240), # appet poor
        np.random.binomial(1, 0.45,240), # pe
        np.random.binomial(1, 0.60,240), # ane
    ])
    X_normal = np.column_stack([
        np.random.normal(45, 12, 160),
        np.random.normal(72, 10, 160),
        np.random.choice([1.005,1.010,1.015,1.020,1.025],160, p=[0.05,0.1,0.25,0.35,0.25]),
        np.random.choice([0,1,2,3,4,5], 160, p=[0.75,0.15,0.06,0.02,0.01,0.01]),
        np.random.choice([0,1,2,3,4,5], 160, p=[0.80,0.12,0.05,0.02,0.005,0.005]),
        np.random.normal(110, 20, 160),
        np.random.normal(30, 10, 160),
        np.random.normal(0.9, 0.2, 160),
        np.random.normal(138, 3, 160),
        np.random.normal(4.3, 0.4, 160),
        np.random.normal(14.5, 1.5, 160),
        np.random.normal(44, 4, 160),
        np.random.normal(7500, 1500,160),
        np.random.normal(4.7, 0.4, 160),
        np.random.binomial(1, 0.05, 160),
        np.random.binomial(1, 0.05, 160),
        np.random.binomial(1, 0.03, 160),
        np.random.binomial(1, 0.02, 160),
        np.random.binomial(1, 0.10, 160),
        np.random.binomial(1, 0.08, 160),
        np.random.binomial(1, 0.02, 160),
        np.random.binomial(1, 0.05, 160),
        np.random.binomial(1, 0.03, 160),
        np.random.binomial(1, 0.05, 160),
    ])
    X = np.vstack([X_ckd, X_normal])
    idx = np.random.permutation(n)
    return X[idx], y[idx]

# ─── Train & evaluate all models ──────────────────────────────────────────────
@st.cache_data
def train_and_evaluate_models():
    X, y = generate_benchmark_data()

    modelos = {
        'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':     GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression':   LogisticRegression(max_iter=500, random_state=42),
        'SVM':                   SVC(probability=True, random_state=42),
        'KNN':                   KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':         DecisionTreeClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred  = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_proba)
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Z-score da acurácia (vs proporção base)
        p0 = y_test.mean()
        se = np.sqrt(p0 * (1 - p0) / len(y_test))
        z_score = (acc - p0) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        resultados[nome] = {
            'modelo':        modelo,
            'y_pred':        y_pred,
            'y_proba':       y_proba,
            'y_test':        y_test,
            'cm':            cm,
            'cv_scores':     cv_scores,
            'Acurácia':      acc,
            'Precisão':      prec,
            'Recall':        rec,
            'F1-Score':      f1,
            'AUC-ROC':       auc,
            'Especificidade':spec,
            'CV Média':      cv_scores.mean(),
            'CV Std':        cv_scores.std(),
            'Z-Score':       z_score,
            'P-Valor':       p_value,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        }

    return resultados

# ─── Sidebar: patient inputs ───────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown('<div class="section-header">🧑‍⚕️ Dados do Paciente</div>', unsafe_allow_html=True)

    age   = st.sidebar.slider('Idade', 2, 90, 50)
    bp    = st.sidebar.slider('Pressão Arterial (bp)', 50, 180, 80)
    sg    = st.sidebar.selectbox('Gravidade Específica (sg)', [1.005, 1.010, 1.015, 1.020, 1.025])
    al    = st.sidebar.slider('Albumina (al)', 0, 5, 0)
    su    = st.sidebar.slider('Açúcar (su)', 0, 5, 0)
    bgr   = st.sidebar.slider('Glicose (bgr)', 22, 490, 120)
    bu    = st.sidebar.slider('Ureia (bu)', 1, 391, 40)
    sc    = st.sidebar.slider('Creatinina (sc)', 0.4, 76.0, 1.2)
    sod   = st.sidebar.slider('Sódio (sod)', 4, 163, 138)
    pot   = st.sidebar.slider('Potássio (pot)', 2.5, 7.6, 4.4)
    hemo  = st.sidebar.slider('Hemoglobina (hemo)', 3.1, 17.8, 13.0)
    pcv   = st.sidebar.slider('Volume Globular (pcv)', 9, 54, 40)
    wc    = st.sidebar.slider('Glóbulos Brancos (wc)', 2200, 26400, 8000)
    rc    = st.sidebar.slider('Glóbulos Vermelhos (rc)', 2.1, 6.5, 4.8)

    st.sidebar.markdown('---')
    rbc   = st.sidebar.selectbox('Glóbulos Vermelhos (rbc)', ['normal', 'abnormal'])
    pc    = st.sidebar.selectbox('Células de Pus (pc)', ['normal', 'abnormal'])
    pcc   = st.sidebar.selectbox('Aglomerados de Pus (pcc)', ['notpresent', 'present'])
    ba    = st.sidebar.selectbox('Bactérias (ba)', ['notpresent', 'present'])
    htn   = st.sidebar.selectbox('Hipertensão (htn)', ['no', 'yes'])
    dm    = st.sidebar.selectbox('Diabetes (dm)', ['no', 'yes'])
    cad   = st.sidebar.selectbox('Doença Coronária (cad)', ['no', 'yes'])
    appet = st.sidebar.selectbox('Apetite (appet)', ['good', 'poor'])
    pe    = st.sidebar.selectbox('Edema (pe)', ['no', 'yes'])
    ane   = st.sidebar.selectbox('Anemia (ane)', ['no', 'yes'])

    return dict(age=age, bp=bp, sg=sg, al=al, su=su, bgr=bgr, bu=bu, sc=sc,
                sod=sod, pot=pot, hemo=hemo, pcv=pcv, wc=wc, rc=rc,
                rbc=rbc, pc=pc, pcc=pcc, ba=ba, htn=htn, dm=dm,
                cad=cad, appet=appet, pe=pe, ane=ane)

# ─── Build patient dataframe ───────────────────────────────────────────────────
def build_patient_df(inp, scaler):
    mapeamento = {
        'rbc':   {'normal': 0, 'abnormal': 1},
        'pc':    {'normal': 0, 'abnormal': 1},
        'pcc':   {'notpresent': 0, 'present': 1},
        'ba':    {'notpresent': 0, 'present': 1},
        'htn':   {'no': 0, 'yes': 1},
        'dm':    {'no': 0, 'yes': 1},
        'cad':   {'no': 0, 'yes': 1},
        'appet': {'good': 0, 'poor': 1},
        'pe':    {'no': 0, 'yes': 1},
        'ane':   {'no': 0, 'yes': 1},
    }
    df = pd.DataFrame([[
        inp['age'], inp['bp'], inp['sg'], inp['al'], inp['su'],
        mapeamento['rbc'][inp['rbc']],
        mapeamento['pc'][inp['pc']],
        mapeamento['pcc'][inp['pcc']],
        mapeamento['ba'][inp['ba']],
        inp['bgr'], inp['bu'], inp['sc'], inp['sod'], inp['pot'],
        inp['hemo'], inp['pcv'], inp['wc'], inp['rc'],
        mapeamento['htn'][inp['htn']],
        mapeamento['dm'][inp['dm']],
        mapeamento['cad'][inp['cad']],
        mapeamento['appet'][inp['appet']],
        mapeamento['pe'][inp['pe']],
        mapeamento['ane'][inp['ane']],
    ]], columns=['age','bp','sg','al','su','rbc','pc','pcc','ba',
                 'bgr','bu','sc','sod','pot','hemo','pcv','wc','rc',
                 'htn','dm','cad','appet','pe','ane'])
    num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# ─── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title='Matriz de Confusão'):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    labels = [['TN\n(Verdadeiro\nNegativo)', 'FP\n(Falso\nPositivo)'],
              ['FN\n(Falso\nNegativo)', 'TP\n(Verdadeiro\nPositivo)']]
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=['Previsto: Não CKD', 'Previsto: CKD'],
                yticklabels=['Real: Não CKD', 'Real: CKD'], ax=ax,
                linewidths=1, linecolor='white', cbar=False)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else '#1a3a5c'
            ax.text(j + 0.5, i + 0.38, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.65, labels[i][j],
                    ha='center', va='center', fontsize=7, color=color)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1a3a5c', pad=10)
    plt.tight_layout()
    return fig

# ─── Plot ROC curves ───────────────────────────────────────────────────────────
def plot_roc_curves(resultados):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ['#2196F3','#4caf50','#f44336','#ff9800','#9c27b0','#00bcd4']
    for (nome, res), cor in zip(resultados.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
        ax.plot(fpr, tpr, lw=2, color=cor,
                label=f"{nome} (AUC={res['AUC-ROC']:.3f})")
    ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Linha Base')
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=10)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=10)
    ax.set_title('Curvas ROC – Comparação de Modelos', fontsize=11, fontweight='bold', color='#1a3a5c')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    return fig

# ─── Plot CV score distributions ──────────────────────────────────────────────
def plot_cv_distributions(resultados):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    nomes  = list(resultados.keys())
    scores = [resultados[n]['cv_scores'] for n in nomes]
    colors = ['#2196F3','#4caf50','#f44336','#ff9800','#9c27b0','#00bcd4']
    bp = ax.boxplot(scores, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2.5))
    for patch, cor in zip(bp['boxes'], colors):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(nomes)+1))
    ax.set_xticklabels([n.replace(' ', '\n') for n in nomes], fontsize=8.5)
    ax.set_ylabel('Acurácia (CV)', fontsize=10)
    ax.set_title('Distribuição da Acurácia – Validação Cruzada (5-fold)', fontsize=11, fontweight='bold', color='#1a3a5c')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    return fig

# ─── Plot Z-Score bar ──────────────────────────────────────────────────────────
def plot_zscore(resultados):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    nomes   = list(resultados.keys())
    zscores = [resultados[n]['Z-Score'] for n in nomes]
    cores   = ['#4caf50' if z > 1.96 else '#ff9800' if z > 1.28 else '#f44336' for z in zscores]
    bars = ax.barh(nomes, zscores, color=cores, edgecolor='white', height=0.55)
    ax.axvline(1.96, color='#2196F3', linestyle='--', linewidth=1.5, label='z=1.96 (p<0.05)')
    ax.axvline(1.28, color='#ff9800', linestyle=':', linewidth=1.5, label='z=1.28 (p<0.10)')
    for bar, z in zip(bars, zscores):
        ax.text(z + 0.05, bar.get_y() + bar.get_height()/2,
                f'{z:.2f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Z-Score da Acurácia', fontsize=10)
    ax.set_title('Z-Score vs Classificador Base (proporção de CKD)', fontsize=11, fontweight='bold', color='#1a3a5c')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    return fig

# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">🫀 Sistema de Diagnóstico de Doença Renal Crónica</div>', unsafe_allow_html=True)
    st.markdown('Introduz os dados do paciente na barra lateral e explora as métricas de avaliação dos modelos.')

    try:
        rf, scaler = load_main_model()
        model_loaded = True
    except Exception:
        model_loaded = False
        st.warning("⚠️ Ficheiros `rf_kidney.pkl` / `scaler_kidney.pkl` não encontrados. "
                   "O diagnóstico individual estará inativo, mas a comparação de modelos funciona.")

    inputs = sidebar_inputs()
    resultados = train_and_evaluate_models()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        '🩺 Diagnóstico Individual',
        '📊 Comparação de Modelos',
        '🔢 Matrizes de Confusão',
        '📈 Análise Estatística (Z-Score)',
    ])

    # ── Tab 1: Diagnóstico ─────────────────────────────────────────────────────
    with tab1:
        if not model_loaded:
            st.info("Carrega os ficheiros `.pkl` para activar o diagnóstico individual.")
        else:
            dados_paciente = build_patient_df(inputs, scaler)
            if st.button('🔍 Prever', type='primary'):
                predicao     = rf.predict(dados_paciente)[0]
                probabilidade = rf.predict_proba(dados_paciente)[0]

                st.markdown('<div class="section-header">Resultado</div>', unsafe_allow_html=True)
                if predicao == 1:
                    st.error('⚠️ Doença Renal Crónica (CKD) Detectada')
                    st.markdown("""**O que é CKD?** A Doença Renal Crónica é uma condição em que os rins
                    perdem progressivamente a capacidade de filtrar o sangue. Se não tratada, pode evoluir para
                    insuficiência renal total.""")
                    st.warning('### 🔴 Consulte um médico nefrologista com urgência.')
                else:
                    st.success('✅ Sem Doença Renal Crónica Detectada')
                    st.info('### 🟢 Continue com acompanhamento médico regular.')

                col1, col2 = st.columns(2)
                col1.metric('Probabilidade CKD',    f'{probabilidade[1]*100:.1f}%')
                col2.metric('Probabilidade Não CKD', f'{probabilidade[0]*100:.1f}%')
                st.progress(float(probabilidade[1]))

                st.markdown('<div class="section-header">Factores de Risco Identificados</div>', unsafe_allow_html=True)
                riscos = []
                if inputs['htn']  == 'yes':      riscos.append('🔴 Hipertensão')
                if inputs['dm']   == 'yes':      riscos.append('🔴 Diabetes')
                if inputs['ane']  == 'yes':      riscos.append('🔴 Anemia')
                if inputs['pe']   == 'yes':      riscos.append('🔴 Edema')
                if inputs['cad']  == 'yes':      riscos.append('🔴 Doença Coronária')
                if inputs['rbc']  == 'abnormal': riscos.append('🔴 Glóbulos Vermelhos Anormais')
                if inputs['pc']   == 'abnormal': riscos.append('🔴 Células de Pus Anormais')
                if inputs['pcc']  == 'present':  riscos.append('🔴 Aglomerados de Pus Presentes')
                if inputs['ba']   == 'present':  riscos.append('🔴 Bactérias Presentes')
                if inputs['appet']== 'poor':     riscos.append('🟡 Apetite Reduzido')
                if inputs['sc']   > 1.5:         riscos.append(f"🔴 Creatinina Elevada ({inputs['sc']})")
                if inputs['hemo'] < 12.0:        riscos.append(f"🟡 Hemoglobina Baixa ({inputs['hemo']})")
                if inputs['pcv']  < 36:          riscos.append(f"🟡 Volume Globular Baixo ({inputs['pcv']})")
                for r in riscos if riscos else ['✅ Nenhum factor de risco identificado.']:
                    st.write(r)

    # ── Tab 2: Comparação de Modelos ───────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📊 Tabela Comparativa de Métricas</div>', unsafe_allow_html=True)

        metricas = ['Acurácia','Precisão','Recall','F1-Score','AUC-ROC','Especificidade','CV Média','CV Std','Z-Score','P-Valor']
        df_comp = pd.DataFrame({nome: {m: res[m] for m in metricas} for nome, res in resultados.items()}).T

        # Identifica melhor modelo por F1
        best = df_comp['F1-Score'].idxmax()
        st.markdown(f'**Melhor Modelo (F1-Score):** <span class="best-model-badge">🏆 {best}</span>', unsafe_allow_html=True)

        # Formatação da tabela
        styled = df_comp.style.format({
            'Acurácia': '{:.3f}', 'Precisão': '{:.3f}', 'Recall': '{:.3f}',
            'F1-Score': '{:.3f}', 'AUC-ROC': '{:.3f}', 'Especificidade': '{:.3f}',
            'CV Média': '{:.3f}', 'CV Std': '{:.4f}', 'Z-Score': '{:.2f}', 'P-Valor': '{:.4f}',
        }).background_gradient(subset=['Acurácia','F1-Score','AUC-ROC'], cmap='Blues') \
          .background_gradient(subset=['Z-Score'], cmap='Greens') \
          .background_gradient(subset=['P-Valor'], cmap='Reds_r') \
          .highlight_max(subset=['Acurácia','Precisão','Recall','F1-Score','AUC-ROC','Especificidade','CV Média'], color='#c8e6c9')
        st.dataframe(styled, use_container_width=True, height=280)

        st.markdown('<div class="section-header">📉 Curvas ROC</div>', unsafe_allow_html=True)
        col_roc, col_cv = st.columns(2)
        with col_roc:
            st.pyplot(plot_roc_curves(resultados))
        with col_cv:
            st.pyplot(plot_cv_distributions(resultados))

        st.markdown('<div class="section-header">📋 Relatório por Modelo</div>', unsafe_allow_html=True)
        modelo_sel = st.selectbox('Seleciona um modelo:', list(resultados.keys()))
        res = resultados[modelo_sel]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Acurácia',   f"{res['Acurácia']:.3f}")
        c2.metric('Precisão',   f"{res['Precisão']:.3f}")
        c3.metric('Recall',     f"{res['Recall']:.3f}")
        c4.metric('F1-Score',   f"{res['F1-Score']:.3f}")
        c5.metric('AUC-ROC',    f"{res['AUC-ROC']:.3f}")
        c6, c7, c8, c9 = st.columns(4)
        c6.metric('CV Média ± Std', f"{res['CV Média']:.3f} ± {res['CV Std']:.4f}")
        c7.metric('Z-Score',        f"{res['Z-Score']:.2f}")
        c8.metric('P-Valor',        f"{res['P-Valor']:.4f}")
        c9.metric('Especificidade', f"{res['Especificidade']:.3f}")

    # ── Tab 3: Matrizes de Confusão ────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🔢 Matrizes de Confusão – Todos os Modelos</div>', unsafe_allow_html=True)
        st.caption("Testado sobre 20% dos dados sintéticos de benchmark (n≈80 amostras).")

        cols = st.columns(3)
        for idx, (nome, res) in enumerate(resultados.items()):
            with cols[idx % 3]:
                fig = plot_confusion_matrix(res['cm'], title=nome)
                st.pyplot(fig)
                plt.close(fig)

                tp, tn = res['TP'], res['TN']
                fp, fn = res['FP'], res['FN']
                total  = tp + tn + fp + fn
                st.markdown(f"""
                <div class="metric-card">
                TP={tp} &nbsp;|&nbsp; TN={tn} &nbsp;|&nbsp; FP={fp} &nbsp;|&nbsp; FN={fn}<br>
                <small>Sensibilidade: {tp/(tp+fn):.2%} &nbsp;|&nbsp; Especificidade: {tn/(tn+fp):.2%}</small>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Análise Estatística ─────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">📈 Z-Score da Acurácia vs Classificador Base</div>', unsafe_allow_html=True)
        st.markdown("""
        O **Z-Score** mede quão significativamente a acurácia de cada modelo supera um classificador
        ingénuo (que prevê sempre a classe maioritária). Valores acima de **1.96** indicam significância
        estatística a p < 0.05.
        """)
        st.pyplot(plot_zscore(resultados))

        st.markdown('<div class="section-header">📐 Tabela de Inferência Estatística</div>', unsafe_allow_html=True)
        df_stat = pd.DataFrame({
            'Modelo':        list(resultados.keys()),
            'Acurácia':      [res['Acurácia'] for res in resultados.values()],
            'Z-Score':       [res['Z-Score']  for res in resultados.values()],
            'P-Valor':       [res['P-Valor']  for res in resultados.values()],
            'Significativo': ['✅ Sim (p<0.05)' if res['P-Valor'] < 0.05
                              else '⚠️ Parcial (p<0.10)' if res['P-Valor'] < 0.10
                              else '❌ Não' for res in resultados.values()],
            'CV Média':      [res['CV Média'] for res in resultados.values()],
            'CV Std':        [res['CV Std']   for res in resultados.values()],
        }).set_index('Modelo')
        st.dataframe(
            df_stat.style.format({
                'Acurácia': '{:.3f}', 'Z-Score': '{:.2f}',
                'P-Valor': '{:.4f}', 'CV Média': '{:.3f}', 'CV Std': '{:.4f}'
            }).background_gradient(subset=['Z-Score'], cmap='Greens')
              .background_gradient(subset=['P-Valor'], cmap='Reds_r'),
            use_container_width=True
        )

        st.markdown('<div class="section-header">ℹ️ Legenda das Métricas</div>', unsafe_allow_html=True)
        st.markdown("""
        | Métrica | Descrição |
        |---|---|
        | **Acurácia** | % de previsões corretas no total |
        | **Precisão** | De todos os CKD previstos, quantos são realmente CKD |
        | **Recall (Sensibilidade)** | De todos os CKD reais, quantos foram detetados |
        | **F1-Score** | Média harmónica de Precisão e Recall |
        | **AUC-ROC** | Área sob a curva ROC (1.0 = perfeito) |
        | **Especificidade** | Capacidade de identificar corretamente os não-CKD |
        | **CV Média ± Std** | Acurácia média e desvio na validação cruzada 5-fold |
        | **Z-Score** | Distância estatística da acurácia ao classificador base |
        | **P-Valor** | Significância estatística (< 0.05 considerado significativo) |
        | **TP/TN/FP/FN** | Verdadeiros/Falsos Positivos e Negativos |
        """)

if __name__ == '__main__':
    main()
