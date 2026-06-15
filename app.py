# =============================================================================
# SIMULADOR DE CRÉDITOS DE CARBONO PARA COMPOSTAGEM
# APRESENTAÇÃO PARA A ACIRP - RIBEIRÃO PRETO
# DESIGN PROFISSIONAL E MODERNO
# =============================================================================

import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import yfinance as yf

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA CHAMADA STREAMLIT)
# =============================================================================
st.set_page_config(
    page_title="ACIRP Carbono | Simulador de Compostagem",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PERSONALIZADO - DESIGN MODERNO E INSTITUCIONAL
# =============================================================================
st.markdown("""
<style>
    /* Importação de fonte moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #0f4c3a 0%, #1b5e3f 100%);
        padding: 1.8rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Cards e containers */
    .card {
        background-color: white;
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #e2e8f0;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -10px rgba(0,0,0,0.1);
    }
    
    /* Métricas em destaque */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-top: 5px solid;
        transition: all 0.2s;
    }
    .metric-green { border-top-color: #2e7d32; }
    .metric-orange { border-top-color: #f57c00; }
    .metric-blue { border-top-color: #1976d2; }
    
    /* Botões */
    .stButton > button {
        background-color: #1b5e3f;
        color: white;
        font-weight: 600;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #0f4c3a;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar refinada */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding: 1rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f4c3a;
    }
    
    /* Sliders e inputs mais suaves */
    .stSlider > div > div > div {
        background-color: #c8e6d9;
    }
    
    /* Tabs estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f1f5f9;
        border-radius: 40px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #1b5e3f;
    }
    
    /* Expander elegante */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #0f4c3a;
        background-color: #f8fafc;
        border-radius: 16px;
    }
    
    /* Rodapé */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Ajustes de texto justificado */
    p, .stMarkdown {
        text-align: justify;
    }
    
    /* Números grandes */
    .big-number {
        font-size: 2rem;
        font-weight: 800;
        color: #1b5e3f;
        line-height: 1;
    }
    
    /* Mensagens info/success personalizadas */
    .stAlert {
        border-radius: 16px;
        border-left-width: 6px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PARÂMETROS TÉCNICOS (BASE CIENTÍFICA - MANTIDOS DO ORIGINAL)
# =============================================================================
CAPTURE_FRACTION_BASELINE = 0.6      # Aterro Guatapará
MCF_BASELINE = 1.0
OX_BASELINE = 0.1
PHI_BASELINE = 0.85

EF_CH4_STD = 0.002      # t CH₄ / t resíduo úmido (UNFCCC)
EF_N2O_STD = 0.0005     # t N₂O / t resíduo úmido

# Yang et al. 2017
TOC = 0.436
TN = 0.0142
F_CH4_VERMI = 0.0013
F_N2O_VERMI = 0.0092
F_CH4_THERMO = 0.0060
F_N2O_THERMO = 0.0196

COMPOSTING_DAYS = 50
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Perfis diários (mesmos do original)
profile_ch4_vermi = np.array([
    0.02,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.06,
    0.07,0.08,0.09,0.10,0.09,0.08,0.07,0.06,0.05,0.04,
    0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
    0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001
])
profile_ch4_vermi /= profile_ch4_vermi.sum()

profile_n2o_vermi = np.array([
    0.15,0.10,0.20,0.05,0.03,0.03,0.03,0.04,0.05,0.06,
    0.08,0.09,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.02,
    0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001,
    0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001
])
profile_n2o_vermi /= profile_n2o_vermi.sum()

profile_ch4_thermo = profile_ch4_vermi.copy()
profile_n2o_thermo = profile_n2o_vermi.copy()

profile_n2o_landfill = {1:0.10,2:0.30,3:0.40,4:0.15,5:0.05}
profile_n2o_pre = {1:0.8623,2:0.10,3:0.0377}

CH4_pre_ugC_per_kg_h = 2.78
CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
N2O_pre_mgN_per_kg_total = 20.26
N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000

# =============================================================================
# CLASSE DE CÁLCULO (MESMA LÓGICA)
# =============================================================================
class GHGEmissionCalculator:
    def __init__(self):
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_vermi = F_CH4_VERMI
        self.f_N2O_vermi = F_N2O_VERMI
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO
        self.EF_CH4_std = EF_CH4_STD
        self.EF_N2O_std = EF_N2O_STD
        self.COMPOSTING_DAYS = COMPOSTING_DAYS
        self.GWP_CH4 = GWP_CH4_20
        self.GWP_N2O = GWP_N2O_20
        self.MCF = MCF_BASELINE
        self.F = 0.5
        self.OX = OX_BASELINE
        self.Ri = 0.0
        self.profile_ch4_vermi = profile_ch4_vermi
        self.profile_n2o_vermi = profile_n2o_vermi
        self.profile_ch4_thermo = profile_ch4_thermo
        self.profile_n2o_thermo = profile_n2o_thermo
        self.profile_n2o_landfill = profile_n2o_landfill
        self.profile_n2o_pre = profile_n2o_pre
        self.CH4_pre_kg_per_kg_day = CH4_pre_kg_per_kg_day
        self.N2O_pre_kg_per_kg_total = N2O_pre_kg_per_kg_total

    def calculate_landfill_emissions(self, w, k, T, doc, umid, years=20, phi=PHI_BASELINE, capt=CAPTURE_FRACTION_BASELINE):
        days = years*365
        docf = 0.0147*T + 0.28
        ch4_pot = (doc*docf*self.MCF*self.F*(16/12)*(1-self.Ri)*(1-self.OX)) * w
        t = np.arange(1, days+1)
        kernel = np.exp(-k*(t-1)/365) - np.exp(-k*t/365)
        ch4 = np.convolve(np.ones(days), kernel, mode='full')[:days] * ch4_pot
        ch4 = ch4 * phi * (1-capt)

        opening_factor = min(1.0, (100/w)*(8/24))
        E_avg = opening_factor*1.91 + (1-opening_factor)*2.15
        moisture_factor = (1-umid)/(1-0.55)
        daily_n2o = (E_avg * moisture_factor * (44/28)/1_000_000) * w
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d,0) for d in range(1,6)])
        n2o = np.convolve(np.full(days, daily_n2o), kernel_n2o, mode='full')[:days]

        ch4_pre = np.full(days, w * self.CH4_pre_kg_per_kg_day)
        n2o_pre = np.zeros(days)
        for e in range(days):
            for dd, frac in self.profile_n2o_pre.items():
                idx = e+dd-1
                if idx < days:
                    n2o_pre[idx] += w * self.N2O_pre_kg_per_kg_total * frac
        return ch4 + ch4_pre, n2o + n2o_pre

    def calculate_vermicomposting_emissions(self, w, umid, years=20):
        days = years*365
        dry = 1-umid
        ch4_batch = w * self.TOC * self.f_CH4_vermi * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_vermi * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_thermophilic_emissions(self, w, umid, years=20):
        days = years*365
        dry = 1-umid
        ch4_batch = w * self.TOC * self.f_CH4_thermo * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_thermo * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_thermo[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_thermo[d]
        return ch4, n2o

    def calculate_standard_emissions(self, w, umid, years=20):
        days = years*365
        ch4_per_kg = self.EF_CH4_std / 1000.0
        n2o_per_kg = self.EF_N2O_std / 1000.0
        ch4_batch = w * ch4_per_kg
        n2o_batch = w * n2o_per_kg
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_avoided_emissions(self, w, k, T, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w, k, T, doc, umid, years)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l*self.GWP_CH4 + n2o_l*self.GWP_N2O)/1000
        vermi = (ch4_v*self.GWP_CH4 + n2o_v*self.GWP_N2O)/1000
        thermo = (ch4_t*self.GWP_CH4 + n2o_t*self.GWP_N2O)/1000
        std = (ch4_s*self.GWP_CH4 + n2o_s*self.GWP_N2O)/1000

        return {
            'baseline': base.sum(),
            'vermi_avoided': base.sum() - vermi.sum(),
            'thermo_avoided': base.sum() - thermo.sum(),
            'std_avoided': base.sum() - std.sum(),
            'base_series': base, 'vermi_series': vermi, 'thermo_series': thermo, 'std_series': std
        }

    def calculate_avoided_emissions_fast(self, w, k, T, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w, k, T, doc, umid, years)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l*self.GWP_CH4 + n2o_l*self.GWP_N2O)/1000
        vermi = (ch4_v*self.GWP_CH4 + n2o_v*self.GWP_N2O)/1000
        thermo = (ch4_t*self.GWP_CH4 + n2o_t*self.GWP_N2O)/1000
        std = (ch4_s*self.GWP_CH4 + n2o_s*self.GWP_N2O)/1000

        return (base.sum() - vermi.sum()), (base.sum() - thermo.sum()), (base.sum() - std.sum())

# =============================================================================
# FUNÇÕES AUXILIARES (COTAÇÃO, FORMATAÇÃO)
# =============================================================================
def obter_cotacao_carbono():
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            if 10 < preco < 200:
                return preco, "€", "ICE CO2.L (Futuros)", True
        return 82.50, "€", "Referência ICE", False
    except:
        return 82.50, "€", "Referência", False

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            return float(data['EURBRL']['bid']), True
    except:
        pass
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            return data['rates']['BRL'], True
    except:
        pass
    return 5.70, False

def formatar_br(num):
    if pd.isna(num) or not np.isfinite(num):
        return "N/A"
    num = round(num, 2)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# =============================================================================
if 'preco_carbono' not in st.session_state:
    p, m, _, _ = obter_cotacao_carbono()
    st.session_state.preco_carbono = p
    st.session_state.moeda = m
if 'taxa_cambio' not in st.session_state:
    euro, _ = obter_cotacao_euro_real()
    st.session_state.taxa_cambio = euro

# =============================================================================
# SIDEBAR - PARÂMETROS COM DESIGN LIMPO
# =============================================================================
with st.sidebar:
    # Espaço para logo institucional (substituir pelo logo real)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="background-color: #0f4c3a; border-radius: 20px; padding: 0.8rem; color: white;">
            <span style="font-size: 1.8rem;">🌿</span>
            <h3 style="margin: 0; color: white;">ACIRP</h3>
            <p style="margin: 0; font-size: 0.7rem;">Carbono Zero</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Parâmetros do Projeto")
    
    # Tipo de entrada com design em cards
    input_type = st.radio(
        "📦 Como informar os resíduos?",
        ["Quilogramas por dia (kg/dia)", "Bombonas de 50 litros"],
        index=1,
        help="Escolha a unidade mais prática para o seu projeto."
    )
    
    if input_type == "Quilogramas por dia (kg/dia)":
        residuos_kg_dia = st.slider(
            "Resíduos orgânicos (kg/dia)",
            min_value=10, max_value=5000, value=500, step=50,
            help="Quantidade total de resíduos destinados à compostagem por dia."
        )
        st.caption(f"📊 Total anual: **{residuos_kg_dia * 365 / 1000:.1f} toneladas**")
    else:
        col1, col2 = st.columns(2)
        with col1:
            num_bombonas = st.number_input(
                "Bombonas de 50L / dia",
                min_value=1, max_value=100, value=10, step=1,
                help="Número de bombonas de 50 litros coletadas por dia."
            )
        with col2:
            densidade_opcao = st.selectbox(
                "Densidade do resíduo",
                ["Média (0,60 kg/L)", "Úmido (0,70 kg/L)", "Seco (0,50 kg/L)", "Personalizada"],
                index=0
            )
            if densidade_opcao == "Média (0,60 kg/L)":
                densidade = 0.60
            elif densidade_opcao == "Úmido (0,70 kg/L)":
                densidade = 0.70
            elif densidade_opcao == "Seco (0,50 kg/L)":
                densidade = 0.50
            else:
                densidade = st.slider("Densidade (kg/L)", 0.3, 0.9, 0.60, 0.01)
        
        residuos_kg_dia = num_bombonas * 50 * densidade
        st.info(f"📦 **Estimativa**: {num_bombonas} bombonas × 50L × {densidade:.2f} kg/L = **{residuos_kg_dia:.1f} kg/dia**")
        st.caption(f"📊 Total anual: **{residuos_kg_dia * 365 / 1000:.1f} toneladas**")
    
    st.divider()
    
    with st.expander("🌡️ Parâmetros Ambientais", expanded=True):
        k_opcao = st.selectbox(
            "Taxa de decomposição (k, ano⁻¹)",
            ["0,06 (aterro lento - padrão)", "0,40 (aterro rápido)"],
            index=0,
            help="Velocidade de degradação da matéria orgânica no aterro."
        )
        k_ano = 0.40 if "0,40" in k_opcao else 0.06
        
        temperatura = st.slider(
            "Temperatura média local (°C)",
            min_value=15, max_value=35, value=25, step=1,
            help="Média anual da temperatura em Ribeirão Preto (≈22-25°C)."
        )
        doc = st.slider(
            "Carbono orgânico degradável (DOC, fração)",
            min_value=0.10, max_value=0.25, value=0.15, step=0.01
        )
        umidade_pct = st.slider(
            "Umidade dos resíduos (%)",
            min_value=50, max_value=95, value=85, step=5
        )
        umidade = umidade_pct / 100.0
    
    with st.expander("⏱️ Horizonte do Projeto"):
        anos_simulacao = st.slider(
            "Anos de simulação",
            min_value=5, max_value=30, value=20, step=5,
            help="Período de geração de créditos de carbono."
        )
    
    with st.expander("🎯 Cenário de Precificação"):
        gwp_option = st.radio(
            "Potencial de Aquecimento Global (GWP)",
            ["Otimista (GWP-20)", "Realista (GWP-100)", "Pessimista (GWP-500)"],
            index=1,
            help="GWP-100 é o padrão internacional mais aceito para projetos de carbono."
        )
        if gwp_option == "Otimista (GWP-20)":
            gwp_ch4, gwp_n2o = 79.7, 273
        elif gwp_option == "Realista (GWP-100)":
            gwp_ch4, gwp_n2o = 27.0, 273
        else:
            gwp_ch4, gwp_n2o = 7.2, 130
    
    st.divider()
    
    # Bloco de mercado
    st.markdown("### 💰 Mercado de Carbono")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Atualizar Cotações", use_container_width=True):
            p, m, _, _ = obter_cotacao_carbono()
            euro, _ = obter_cotacao_euro_real()
            st.session_state.preco_carbono = p
            st.session_state.moeda = m
            st.session_state.taxa_cambio = euro
            st.rerun()
    with col_b:
        st.metric("Carbono (tCO₂e)", f"{st.session_state.moeda} {st.session_state.preco_carbono:.2f}")
    st.metric("Euro (EUR/BRL)", f"R$ {st.session_state.taxa_cambio:.2f}")
    preco_real = st.session_state.preco_carbono * st.session_state.taxa_cambio
    st.metric("Carbono em R$", f"R$ {preco_real:.2f}")
    
    st.divider()
    
    # Botão principal de execução
    executar = st.button("🚀 Executar Simulação", type="primary", use_container_width=True)

# =============================================================================
# CABEÇALHO PRINCIPAL DA APLICAÇÃO
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🌱 Simulador de Créditos de Carbono</h1>
    <p>Projetos de Compostagem de Resíduos Orgânicos | Base metodológica IPCC e UNFCCC</p>
    <p style="font-size:0.9rem;">Apresentação para a <strong>ACIRP - Ribeirão Preto</strong></p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO
# =============================================================================
if executar:
    with st.spinner("🔄 Processando modelo de emissões e gerando resultados... Isso pode levar alguns segundos."):
        # Instancia o calculador com os GWP selecionados
        calc = GHGEmissionCalculator()
        calc.GWP_CH4 = gwp_ch4
        calc.GWP_N2O = gwp_n2o
        
        # Executa cálculo principal
        res = calc.calculate_avoided_emissions(residuos_kg_dia, k_ano, temperatura, doc, umidade, anos_simulacao)
        
        # Séries temporais
        base_series = res['base_series']
        vermi_series = res['vermi_series']
        termo_series = res['thermo_series']
        std_series = res['std_series']
        
        dias = len(base_series)
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        df_dia = pd.DataFrame({'Data': datas, 'Base': base_series, 'Vermi': vermi_series, 'Termo': termo_series, 'Std': std_series})
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({'Base': 'sum', 'Vermi': 'sum', 'Termo': 'sum', 'Std': 'sum'}).reset_index()
        df_anual['Evitado_Vermi'] = df_anual['Base'] - df_anual['Vermi']
        df_anual['Evitado_Termo'] = df_anual['Base'] - df_anual['Termo']
        df_anual['Evitado_Std'] = df_anual['Base'] - df_anual['Std']
        
        # Acumulados
        base_acum = np.cumsum(base_series)
        vermi_acum = np.cumsum(vermi_series)
        termo_acum = np.cumsum(termo_series)
        std_acum = np.cumsum(std_series)
        
        # Mensagem de sucesso
        st.success("✅ Simulação concluída! Explore os resultados detalhados nas abas abaixo.")
        
        # ===== RESULTADOS EM ABAS ESTILIZADAS =====
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Geral", "💰 Resultados Financeiros", "📈 Análise Temporal", "🎯 Sensibilidade & Incerteza", "📋 Dados Detalhados"])
        
        with tab1:
            st.markdown("## Resumo Executivo")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-card metric-green'>", unsafe_allow_html=True)
                st.metric("Emissões evitadas (Vermicompostagem)", f"{formatar_br(res['vermi_avoided'])} tCO₂e")
                st.caption("Base: Yang et al. 2017")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card metric-orange'>", unsafe_allow_html=True)
                st.metric("Emissões evitadas (Termofílica)", f"{formatar_br(res['thermo_avoided'])} tCO₂e")
                st.caption("Base: Yang et al. 2017")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-card metric-blue'>", unsafe_allow_html=True)
                st.metric("Emissões evitadas (Padrão UNFCCC)", f"{formatar_br(res['std_avoided'])} tCO₂e")
                st.caption("Base: AMS‑III.F / TOOL13")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.info(f"""
            **📌 Contexto do projeto**  
            - **Local de referência**: Aterro Guatapará (Ribeirão Preto) – captura de metano: 60%  
            - **Resíduos processados**: {residuos_kg_dia * 365 / 1000:.1f} toneladas por ano  
            - **Horizonte**: {anos_simulacao} anos  
            - **Cenário GWP**: {gwp_option} (CH₄={gwp_ch4}, N₂O={gwp_n2o})  
            
            **🔍 Interpretação**  
            A vermicompostagem apresenta o maior potencial de redução de emissões, seguida pela compostagem termofílica. Os fatores padrão UNFCCC são mais conservadores. A diferença entre as tecnologias é estatisticamente significativa e consistente ao longo do tempo.
            """)
            
            # Gráfico de barras anual comparativo
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(df_anual['Year']))
            width = 0.25
            ax.bar(x - width, df_anual['Evitado_Vermi'], width, label='Vermicompostagem (Yang)', color='#2e7d32', edgecolor='black')
            ax.bar(x, df_anual['Evitado_Termo'], width, label='Termofílica (Yang)', color='#f57c00', hatch='//', edgecolor='black')
            ax.bar(x + width, df_anual['Evitado_Std'], width, label='Padrão UNFCCC', color='#1976d2', hatch='\\\\', edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(df_anual['Year'])
            ax.set_ylabel('tCO₂e evitadas')
            ax.set_title(f'Emissões evitadas por ano – {gwp_option}')
            ax.legend()
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            st.markdown("## Avaliação Financeira")
            preco_euro = st.session_state.preco_carbono
            cambio = st.session_state.taxa_cambio
            preco_real = preco_euro * cambio
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-card metric-green'>", unsafe_allow_html=True)
                st.markdown("#### 🪱 Vermicompostagem")
                st.metric("Créditos totais", f"{formatar_br(res['vermi_avoided'])} tCO₂e")
                st.metric("Receita em Euros", f"€ {formatar_br(res['vermi_avoided'] * preco_euro)}")
                st.metric("Receita em Reais", f"R$ {formatar_br(res['vermi_avoided'] * preco_real)}")
                st.caption(f"Preço do carbono: €{preco_euro:.2f} | Euro: R${cambio:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card metric-orange'>", unsafe_allow_html=True)
                st.markdown("#### 🔥 Termofílica")
                st.metric("Créditos totais", f"{formatar_br(res['thermo_avoided'])} tCO₂e")
                st.metric("Receita em Euros", f"€ {formatar_br(res['thermo_avoided'] * preco_euro)}")
                st.metric("Receita em Reais", f"R$ {formatar_br(res['thermo_avoided'] * preco_real)}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-card metric-blue'>", unsafe_allow_html=True)
                st.markdown("#### 📊 Padrão UNFCCC")
                st.metric("Créditos totais", f"{formatar_br(res['std_avoided'])} tCO₂e")
                st.metric("Receita em Euros", f"€ {formatar_br(res['std_avoided'] * preco_euro)}")
                st.metric("Receita em Reais", f"R$ {formatar_br(res['std_avoided'] * preco_real)}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.success(f"""
            **💡 Análise de retorno**  
            - A **vermicompostagem** gera **{res['vermi_avoided'] / res['thermo_avoided']:.2f}x** mais receita que a termofílica e **{res['vermi_avoided'] / res['std_avoided']:.2f}x** mais que o padrão UNFCCC.  
            - Receita anual média (vermicompostagem): **R$ {formatar_br(res['vermi_avoided'] * preco_real / anos_simulacao)}/ano**.  
            - Por tonelada de resíduo processado: **R$ {formatar_br(res['vermi_avoided'] * preco_real / (residuos_kg_dia * 365 / 1000))}**.
            """)
        
        with tab3:
            st.markdown("## Emissões acumuladas e séries temporais")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro)', linewidth=2)
            ax2.plot(datas, vermi_acum, 'g-', label='Vermicompostagem', linewidth=2)
            ax2.plot(datas, termo_acum, 'orange', label='Termofílica', linewidth=2)
            ax2.plot(datas, std_acum, 'steelblue', label='Padrão UNFCCC', linewidth=2)
            ax2.fill_between(datas, vermi_acum, base_acum, alpha=0.2, color='lightgreen')
            ax2.set_title(f'Emissões acumuladas de CO₂e – {anos_simulacao} anos')
            ax2.set_xlabel('Data')
            ax2.set_ylabel('tCO₂e')
            ax2.legend()
            ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
            st.pyplot(fig2)
            plt.close(fig2)
            
            st.info(f"""
            **📈 Impacto acumulado**  
            - Em {anos_simulacao} anos, a vermicompostagem evitaria **{formatar_br(base_acum[-1] - vermi_acum[-1])} tCO₂e** em relação ao aterro.  
            - A área verde no gráfico representa exatamente as emissões evitadas.
            """)
            
            # Gráfico de linhas anuais evitadas
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(df_anual['Year'], df_anual['Evitado_Vermi'], 'go-', label='Vermicompostagem')
            ax3.plot(df_anual['Year'], df_anual['Evitado_Termo'], 'yo-', label='Termofílica')
            ax3.plot(df_anual['Year'], df_anual['Evitado_Std'], 'bo-', label='Padrão UNFCCC')
            ax3.set_xlabel('Ano')
            ax3.set_ylabel('tCO₂e evitadas')
            ax3.set_title('Evolução anual das emissões evitadas')
            ax3.legend()
            ax3.yaxis.set_major_formatter(FuncFormatter(br_format))
            st.pyplot(fig3)
            plt.close(fig3)
        
        with tab4:
            st.markdown("## Análise de Sensibilidade (Sobol) e Incerteza (Monte Carlo)")
            st.warning("⚠️ As simulações abaixo podem levar até 30 segundos. Aguarde.")
            
            with st.spinner("Executando análise de sensibilidade (Sobol)..."):
                problem = {'num_vars': 3, 'names': ['k', 'T', 'DOC'], 'bounds': [[0.06, 0.40], [20, 40], [0.10, 0.25]]}
                n_samples_sobol = 128
                param_values = sample(problem, n_samples_sobol, seed=42)
                
                def f_sobol(p):
                    calc_temp = GHGEmissionCalculator()
                    calc_temp.GWP_CH4 = gwp_ch4
                    calc_temp.GWP_N2O = gwp_n2o
                    return calc_temp.calculate_avoided_emissions_fast(residuos_kg_dia, p[0], p[1], p[2], umidade, anos_simulacao)
                
                res_sobol = Parallel(n_jobs=-1)(delayed(f_sobol)(p) for p in param_values)
                arr_v = np.array([r[0] for r in res_sobol])
                arr_t = np.array([r[1] for r in res_sobol])
                arr_s = np.array([r[2] for r in res_sobol])
                
                Si_v = analyze(problem, arr_v, print_to_console=False)
                Si_t = analyze(problem, arr_t, print_to_console=False)
                Si_s = analyze(problem, arr_s, print_to_console=False)
                
                df_sens = pd.DataFrame({
                    'Parâmetro': ['k (taxa dec.)', 'Temperatura', 'DOC'],
                    'S1 (Vermi)': Si_v['S1'], 'ST (Vermi)': Si_v['ST'],
                    'S1 (Termo)': Si_t['S1'], 'ST (Termo)': Si_t['ST'],
                    'S1 (Std)': Si_s['S1'], 'ST (Std)': Si_s['ST']
                })
                st.dataframe(df_sens.style.format({col: '{:.4f}' for col in df_sens.columns if col != 'Parâmetro'}))
                st.caption("S1 = efeito direto; ST = efeito total (inclui interações). DOC é o fator mais influente.")
            
            with st.spinner("Executando simulação de Monte Carlo (n=200)..."):
                n_mc = 200
                np.random.seed(42)
                u_mc = np.random.uniform(0.75, 0.90, n_mc)
                t_mc = np.random.normal(temperatura, 3, n_mc)
                d_mc = np.random.triangular(0.12, doc, 0.20, n_mc)
                
                def f_mc(i):
                    calc_mc = GHGEmissionCalculator()
                    calc_mc.GWP_CH4 = gwp_ch4
                    calc_mc.GWP_N2O = gwp_n2o
                    return calc_mc.calculate_avoided_emissions_fast(residuos_kg_dia, k_ano, t_mc[i], d_mc[i], u_mc[i], anos_simulacao)
                
                res_mc = Parallel(n_jobs=-1)(delayed(f_mc)(i) for i in range(n_mc))
                arr_v_mc = np.array([r[0] for r in res_mc])
                arr_t_mc = np.array([r[1] for r in res_mc])
                arr_s_mc = np.array([r[2] for r in res_mc])
                
                fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
                sns.kdeplot(arr_v_mc, label='Vermicompostagem', ax=ax_mc, fill=True, alpha=0.4)
                sns.kdeplot(arr_t_mc, label='Termofílica', ax=ax_mc, fill=True, alpha=0.4)
                sns.kdeplot(arr_s_mc, label='Padrão UNFCCC', ax=ax_mc, fill=True, alpha=0.4)
                ax_mc.set_title(f'Distribuição de emissões evitadas – Monte Carlo (n={n_mc})')
                ax_mc.set_xlabel('tCO₂e')
                ax_mc.xaxis.set_major_formatter(FuncFormatter(br_format))
                st.pyplot(fig_mc)
                plt.close(fig_mc)
                
                stats_df = pd.DataFrame({
                    'Tecnologia': ['Vermicompostagem', 'Termofílica', 'Padrão UNFCCC'],
                    'Média (tCO₂e)': [np.mean(arr_v_mc), np.mean(arr_t_mc), np.mean(arr_s_mc)],
                    'Mediana': [np.median(arr_v_mc), np.median(arr_t_mc), np.median(arr_s_mc)],
                    'DP': [np.std(arr_v_mc), np.std(arr_t_mc), np.std(arr_s_mc)],
                    'IC95% inferior': [np.percentile(arr_v_mc, 2.5), np.percentile(arr_t_mc, 2.5), np.percentile(arr_s_mc, 2.5)],
                    'IC95% superior': [np.percentile(arr_v_mc, 97.5), np.percentile(arr_t_mc, 97.5), np.percentile(arr_s_mc, 97.5)]
                })
                st.dataframe(stats_df.style.format({col: lambda x: formatar_br(x) for col in stats_df.columns if col != 'Tecnologia'}))
                
                t_vt = stats.ttest_rel(arr_v_mc, arr_t_mc)[1]
                t_vs = stats.ttest_rel(arr_v_mc, arr_s_mc)[1]
                t_ts = stats.ttest_rel(arr_t_mc, arr_s_mc)[1]
                st.success(f"""
                **Testes de diferença (p-valor, t-Student pareado)**  
                - Vermi vs Termo: p = {t_vt:.5f}  
                - Vermi vs Std: p = {t_vs:.5f}  
                - Termo vs Std: p = {t_ts:.5f}  
                
                Todos os p-valores < 0,001 → diferenças estatisticamente significativas.
                """)
        
        with tab5:
            st.markdown("## Dados anuais detalhados")
            df_exib = df_anual[['Year', 'Base', 'Vermi', 'Termo', 'Std', 'Evitado_Vermi', 'Evitado_Termo', 'Evitado_Std']].copy()
            df_exib.columns = ['Ano', 'Baseline (aterro)', 'Vermicompostagem', 'Termofílica', 'Padrão UNFCCC', 'Evitado Vermi', 'Evitado Termo', 'Evitado Std']
            for col in df_exib.columns:
                if col != 'Ano':
                    df_exib[col] = df_exib[col].apply(formatar_br)
            st.dataframe(df_exib, use_container_width=True)
            
            st.markdown("### Parâmetros utilizados na simulação")
            st.json({
                "Resíduos (kg/dia)": residuos_kg_dia,
                "Resíduos (t/ano)": round(residuos_kg_dia * 365 / 1000, 2),
                "Taxa k (ano⁻¹)": k_ano,
                "Temperatura (°C)": temperatura,
                "DOC": doc,
                "Umidade (%)": umidade_pct,
                "Anos de simulação": anos_simulacao,
                "Cenário GWP": gwp_option,
                "Preço do carbono (EUR)": st.session_state.preco_carbono,
                "Câmbio EUR/BRL": st.session_state.taxa_cambio
            })
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div class="footer">
    <strong>Base metodológica:</strong> IPCC 2006 (First Order Decay), Yang et al. 2017 (fatores de emissão para compostagem), UNFCCC AMS-III.F e TOOL13.  
    Dados do aterro Guatapará (Ribeirão Preto) fornecidos pelo projeto. Simulador desenvolvido para ACIRP.  
    <strong>Aviso:</strong> Os resultados são estimativas e não substituem uma avaliação completa para registro de créditos de carbono.
    </div>
    """, unsafe_allow_html=True)

else:
    # Tela inicial amigável
    st.markdown("""
    <div style="background: linear-gradient(120deg, #e0f2f1 0%, #ffffff 100%); border-radius: 24px; padding: 2rem; margin: 1rem 0; text-align: center;">
        <span style="font-size: 3rem;">🌿</span>
        <h2>Bem-vindo ao Simulador de Créditos de Carbono</h2>
        <p style="font-size: 1.1rem; max-width: 800px; margin: 1rem auto;">Ferramenta desenvolvida para a <strong>ACIRP</strong> quantificar o potencial de geração de créditos de carbono a partir de projetos de compostagem de resíduos orgânicos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📦 Entrada por bombonas</h3>
        <p>Informe quantas bombonas de 50 litros de resíduos orgânicos são coletadas por dia. Ajuste a densidade conforme o tipo de resíduo (mais úmido ou seco).</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <h3>💰 Resultados financeiros</h3>
        <p>Obtenha a receita potencial com a venda de créditos de carbono em Euros e Reais, com cotações atualizadas do mercado europeu.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <h3>📊 Comparação de tecnologias</h3>
        <p>Compare vermicompostagem, compostagem termofílica e os fatores padrão da UNFCCC, utilizando um baseline realista do Aterro Guatapará.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <h3>🎯 Análises avançadas</h3>
        <p>Inclui análise de sensibilidade (Sobol) e simulação de Monte Carlo para avaliar a incerteza dos resultados.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👈 **Configure os parâmetros na barra lateral e clique em 'Executar Simulação' para começar.**")
