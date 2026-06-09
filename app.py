import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar modelo e scaler
rf = joblib.load('rf_kidney.pkl')
scaler = joblib.load('scaler_kidney.pkl')

st.title('Sistema de Diagnóstico de Doença Renal Crónica')
st.markdown('Introduz os dados do paciente para prever se tem CKD ou não.')

st.sidebar.header('Dados do Paciente')

# Features numéricas
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

# Features categóricas
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

# Mapeamento categóricas
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

# Construir dataframe do paciente na ordem correcta
dados_paciente = pd.DataFrame([[
    age, bp, sg, al, su,
    mapeamento['rbc'][rbc],
    mapeamento['pc'][pc],
    mapeamento['pcc'][pcc],
    mapeamento['ba'][ba],
    bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
    mapeamento['htn'][htn],
    mapeamento['dm'][dm],
    mapeamento['cad'][cad],
    mapeamento['appet'][appet],
    mapeamento['pe'][pe],
    mapeamento['ane'][ane]
]], columns=['age','bp','sg','al','su','rbc','pc','pcc','ba',
             'bgr','bu','sc','sod','pot','hemo','pcv','wc','rc',
             'htn','dm','cad','appet','pe','ane'])

# Normalizar apenas as numéricas
num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
dados_paciente[num_cols] = scaler.transform(dados_paciente[num_cols])
# Prever
if st.button('Prever'):
    predicao = rf.predict(dados_paciente)[0]
    probabilidade = rf.predict_proba(dados_paciente)[0]

    st.subheader('Resultado')

    if predicao == 1:
        st.error('⚠️ Doença Renal Crónica (CKD) Detectada')
        st.markdown("""
        **O que é CKD?**
        A Doença Renal Crónica (CKD) é uma condição em que os rins perdem progressivamente 
        a capacidade de filtrar o sangue. Se não tratada, pode evoluir para insuficiência renal total.
        """)
        st.markdown('### 🔴 Recomendação')
        st.warning('Consulte um médico nefrologista com urgência para confirmação e tratamento.')
    else:
        st.success('✅ Sem Doença Renal Crónica Detectada')
        st.markdown("""
        **Resultado Normal**
        Com base nos dados introduzidos, o modelo não detectou sinais de Doença Renal Crónica.
        Mantenha hábitos saudáveis e faça exames de rotina regularmente.
        """)
        st.markdown('### 🟢 Recomendação')
        st.info('Continue com acompanhamento médico regular e mantenha um estilo de vida saudável.')

    # Barra de probabilidade visual
    st.subheader('Probabilidades')
    col1, col2 = st.columns(2)
    col1.metric('CKD', f'{probabilidade[1]*100:.1f}%')
    col2.metric('Não CKD', f'{probabilidade[0]*100:.1f}%')

    st.progress(float(probabilidade[1]))

    # Factores de risco do paciente
    st.subheader('Factores de Risco Identificados')
    riscos = []
    if htn == 'yes':     riscos.append('🔴 Hipertensão')
    if dm == 'yes':      riscos.append('🔴 Diabetes')
    if ane == 'yes':     riscos.append('🔴 Anemia')
    if pe == 'yes':      riscos.append('🔴 Edema')
    if cad == 'yes':     riscos.append('🔴 Doença Coronária')
    if rbc == 'abnormal': riscos.append('🔴 Glóbulos Vermelhos Anormais')
    if pc == 'abnormal':  riscos.append('🔴 Células de Pus Anormais')
    if pcc == 'present':  riscos.append('🔴 Aglomerados de Pus Presentes')
    if ba == 'present':   riscos.append('🔴 Bactérias Presentes')
    if appet == 'poor':   riscos.append('🟡 Apetite Reduzido')
    if sc > 1.5:          riscos.append(f'🔴 Creatinina Elevada ({sc})')
    if hemo < 12.0:       riscos.append(f'🟡 Hemoglobina Baixa ({hemo})')
    if pcv < 36:          riscos.append(f'🟡 Volume Globular Baixo ({pcv})')

    if riscos:
        for r in riscos:
            st.write(r)
    else:
        st.write('✅ Nenhum factor de risco identificado.')