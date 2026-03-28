# =====================================================================
# generate_monthly_maps_fixed.py (VERSÃO CORRIGIDA - SEM LATEX)
# =====================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Importações do seu projeto
from dataset import SPIDataset
from utils_data import load_grid_data, calculate_spi, check_leakage
from model_convlstm3d import ConvLSTM3D
from visualization_spi_classes import set_journal_style

# ==============================================================
# CONFIGURAÇÕES
# ==============================================================
DATA_PATH = "data/pr_Area2.xlsx" #--MUDAR AQUI!
MODEL_PATH = "EXPERIMENTS/P3_Q1/ConvLSTM3D/best_model.pt" #--MUDAR AQUI!
SPI_SCALE = 3
SPLIT_DATE = "2018-01-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Período de interesse: últimas 12 datas do período de VALIDAÇÃO
N_MONTHS = 12
HORIZONTE_EXIBIR = 1

# Configurações de plotagem
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.3,
})

# ==============================================================
# CARREGAR DADOS (SEM RECALCULAR SPI)
# ==============================================================
print("=" * 60)
print("CARREGANDO DADOS (SEM RECALCULAR SPI)")
print("=" * 60)

df_pr = load_grid_data(DATA_PATH)

# Verificar se já existe SPI calculado
spi_cache_path = Path("EXPERIMENTS/df_spi_calculado.pkl") #--MUDAR AQUI!

if spi_cache_path.exists():
    print(f"Carregando SPI de cache: {spi_cache_path}")
    df_spi = pd.read_pickle(spi_cache_path)
else:
    print("Calculando SPI com split temporal correto...")
    df_spi = calculate_spi(
        df_pr,
        scale=SPI_SCALE,
        split_date=SPLIT_DATE,
    )
    check_leakage(df_pr, df_spi, SPLIT_DATE, scale=SPI_SCALE)
    df_spi.to_pickle(spi_cache_path)
    print(f"SPI salvo em {spi_cache_path}")

# Identificar períodos
dates = pd.to_datetime(df_pr.columns)
sd = pd.to_datetime(SPLIT_DATE)
idx_split = next((i for i, d in enumerate(dates) if d >= sd), len(dates))

print(f"\nPeríodo completo: {dates[0].date()} a {dates[-1].date()}")
print(f"Split date: {SPLIT_DATE}")
print(f"  Treino: {dates[0].date()} a {dates[idx_split-1].date()}")
print(f"  Validação: {dates[idx_split].date()} a {dates[-1].date()}")

# ==============================================================
# CARREGAR MODELO
# ==============================================================
print("\n" + "=" * 60)
print("CARREGANDO MODELO")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
P = checkpoint['P']
Q = checkpoint['Q']
hidden = checkpoint['hidden']

print(f"Modelo: P={P}, Q={Q}, hidden={hidden}")
print(f"Melhor WI: {checkpoint.get('best_wi', 'N/A')}")

model = ConvLSTM3D(hidden=hidden).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==============================================================
# SELECIONAR DATAS DE INTERESSE
# ==============================================================
print("\n" + "=" * 60)
print("SELECIONANDO DATAS PARA VISUALIZAÇÃO")
print("=" * 60)

datas_validacao = dates[idx_split:]

if len(datas_validacao) == 0:
    raise ValueError("Não há dados de validação! Verifique SPLIT_DATE.")

datas_interesse = datas_validacao[-min(N_MONTHS, len(datas_validacao)):]

print(f"Total de meses na validação: {len(datas_validacao)}")
print(f"Período selecionado: {datas_interesse[0].date()} a {datas_interesse[-1].date()}")

# ==============================================================
# EXTRAIR LAT/LON
# ==============================================================
lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))
H, W = len(lats), len(lons)

print(f"Grid: {H}×{W} pixels")

# ==============================================================
# FUNÇÃO PARA OBTER ENTRADA E ALVO
# ==============================================================
def get_input_target(data_alvo):
    """
    Para uma data específica, retorna:
        x_tensor: [1, P, 3, H, W] (entrada para o modelo)
        spi_real: [H, W] (SPI observado na data alvo)
    """
    idx_alvo = list(dates).index(data_alvo)
    idx_inicio = idx_alvo - P
    
    if idx_inicio < 0:
        raise ValueError(f"Não há dados suficientes antes de {data_alvo} (P={P})")
    
    # Extrair valores
    x_pr = np.full((P, H, W), np.nan, dtype=np.float32)
    x_spi = np.full((P, H, W), np.nan, dtype=np.float32)
    
    for t, idx_t in enumerate(range(idx_inicio, idx_alvo)):
        data_t = dates[idx_t]
        col_pr = df_pr[data_t].values.reshape(H, W)
        col_spi = df_spi[data_t].values.reshape(H, W)
        x_pr[t] = col_pr
        x_spi[t] = col_spi
    
    # Calcular delta SPI
    x_dspi = np.zeros_like(x_spi)
    if P > 1:
        x_dspi[1:] = x_spi[1:] - x_spi[:-1]
    
    # Stack e tratar NaNs
    x = np.stack([x_pr, x_spi, x_dspi], axis=1)
    x = np.nan_to_num(x, nan=0.0)
    
    # SPI real
    spi_real = df_spi[data_alvo].values.reshape(H, W)
    
    return (
        torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        spi_real
    )

# ==============================================================
# FUNÇÃO PARA OBTER PREDIÇÃO (CORRIGIDA - COM DETACH)
# ==============================================================
def get_prediction(x_tensor):
    """Retorna predição como numpy array (desanexada do grafo)"""
    with torch.no_grad():  # Garantir que não há gradiente
        if HORIZONTE_EXIBIR == 1:
            spi_pred = model.forward_one_step(x_tensor)
        else:
            pred_seq = model.forecast(x_tensor, Q)
            spi_pred = pred_seq[:, HORIZONTE_EXIBIR - 1, :, :]
        
        # Converter para numpy com detach
        spi_pred_np = spi_pred.detach().cpu().numpy().squeeze()
    
    return spi_pred_np

# ==============================================================
# FUNÇÃO PARA CRIAR FIGURA (VERSÃO PDF - SEM LATEX)
# ==============================================================
def create_figure_journal(observado, previsto, data, lats, lons, horizonte):
    """
    Cria figura no formato journal: observado | previsto | barra
    Versão para PDF sem LaTeX - sem títulos, apenas eixos e barra
    """
    img_height = len(lats)
    img_width = len(lons)
    aspect_ratio = img_width / img_height
    
    base_height = 4
    base_width = base_height * aspect_ratio
    
    fig = plt.figure(figsize=(base_width * 2.2, base_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)
    
    # Observado
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(observado, cmap='RdBu', vmin=-3, vmax=3,
                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                     aspect='auto')
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    # Previsto
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(previsto, cmap='RdBu', vmin=-3, vmax=3,
                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                     aspect='auto')
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # Barra de cores
    cax = fig.add_subplot(gs[2])
    cbar = plt.colorbar(im1, cax=cax, label='SPI')
    cbar.ax.tick_params(labelsize=8)
    
    return fig

# ==============================================================
# GERAR MAPAS INDIVIDUAIS (PDF)
# ==============================================================
print("\n" + "=" * 60)
print("GERANDO MAPAS INDIVIDUAIS (PDF)")
print("=" * 60)

out_dir = Path("mapas_mensais_fixed_pdf")
out_dir.mkdir(exist_ok=True)

set_journal_style()

monthly_metrics = {}
all_predictions = {}  # Guardar para o resumo

with torch.no_grad():  # Contexto global sem gradiente
    for i, data in enumerate(datas_interesse):
        print(f"Processando {i+1}/{len(datas_interesse)}: {data.date()}...")
        
        try:
            # Obter entrada e real
            x_tensor, spi_real = get_input_target(data)
            
            # Obter predição (agora com detach interno)
            spi_pred = get_prediction(x_tensor)
            
            # Guardar para o resumo
            all_predictions[data] = {
                'real': spi_real,
                'pred': spi_pred
            }
            
            # Verificar dados válidos
            if np.all(np.isnan(spi_real)) or np.all(np.isnan(spi_pred)):
                print(f"  ⚠ Dados inválidos para {data.date()}, pulando...")
                continue
            
            # Calcular métricas
            mask = ~np.isnan(spi_real) & ~np.isnan(spi_pred)
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((spi_real[mask] - spi_pred[mask])**2))
                mae = np.mean(np.abs(spi_real[mask] - spi_pred[mask]))
                monthly_metrics[data.date()] = {
                    'rmse': float(rmse),
                    'mae': float(mae)
                }
            
            # Criar e salvar figura em PDF
            fig = create_figure_journal(
                spi_real, spi_pred, data, lats, lons, HORIZONTE_EXIBIR
            )
            
            fname = out_dir / f"SPI3_{data.date()}_h{HORIZONTE_EXIBIR}.pdf"
            plt.savefig(fname, dpi=300, bbox_inches='tight', format='pdf')
            plt.close(fig)
            
            print(f"  ✅ Salvo: {fname.name}")
            
        except Exception as e:
            print(f"  ❌ Erro em {data.date()}: {e}")
            continue

# ==============================================================
# SALVAR MÉTRICAS
# ==============================================================
if monthly_metrics:
    metrics_df = pd.DataFrame.from_dict(monthly_metrics, orient='index')
    metrics_df.index.name = 'date'
    metrics_df.to_csv(out_dir / f"monthly_metrics_h{HORIZONTE_EXIBIR}.csv")
    print(f"\n✅ Métricas mensais salvas")
    print(metrics_df.round(4))

# ==============================================================
# GERAR FIGURA RESUMO (PDF)
# ==============================================================
print("\n" + "=" * 60)
print("GERANDO FIGURA RESUMO (PDF)")
print("=" * 60)

if len(all_predictions) == 0:
    print("❌ Nenhuma predição válida para gerar resumo")
else:
    n_months = len(all_predictions)
    n_cols = 4
    n_rows = (n_months + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten() if n_months > 1 else [axes]
    
    # Variável para guardar a última imagem para a colorbar
    last_im = None
    
    for idx, (data, dados) in enumerate(all_predictions.items()):
        try:
            spi_real = dados['real']
            spi_pred = dados['pred']
            
            # Observado
            ax_obs = axes[idx * 2]
            im_obs = ax_obs.imshow(spi_real, cmap='RdBu', vmin=-3, vmax=3,
                                   extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                                   aspect='auto')
            ax_obs.set_title(f"{data.date()}", fontsize=10)
            ax_obs.set_xticks([])
            ax_obs.set_yticks([])
            ax_obs.grid(False)
            
            # Previsto
            ax_pred = axes[idx * 2 + 1]
            im_pred = ax_pred.imshow(spi_pred, cmap='RdBu', vmin=-3, vmax=3,
                                     extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                                     aspect='auto')
            ax_pred.set_title(f"{data.date()}", fontsize=10)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            ax_pred.grid(False)
            
            last_im = im_obs  # Guardar referência para colorbar
            
        except Exception as e:
            print(f"Erro no resumo para {data.date()}: {e}")
            continue
    
    # Esconder subplots não utilizados
    for idx in range(len(all_predictions) * 2, len(axes)):
        axes[idx].set_visible(False)
    
    # Barra de cores (só se tivemos pelo menos uma imagem)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax, label='SPI')
        cbar.ax.tick_params(labelsize=8)
        
        # Sem título superior, apenas ajuste de layout
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        resumo_path = out_dir / f"resumo_{datas_interesse[0].date()}_a_{datas_interesse[-1].date()}_h{HORIZONTE_EXIBIR}.pdf"
        plt.savefig(resumo_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ Figura resumo salva: {resumo_path}")
    else:
        print("❌ Não foi possível gerar a figura resumo")
    
    plt.close(fig)

print(f"\nTodos os arquivos salvos em: {out_dir.resolve()}")

# ==============================================================
# ESTATÍSTICAS RESUMO
# ==============================================================
if monthly_metrics:
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS RESUMO")
    print("=" * 60)
    
    metrics_array = np.array([(v['rmse'], v['mae']) for v in monthly_metrics.values()])
    
    print(f"Média RMSE: {np.mean(metrics_array[:, 0]):.4f} ± {np.std(metrics_array[:, 0]):.4f}")
    print(f"Média MAE:  {np.mean(metrics_array[:, 1]):.4f} ± {np.std(metrics_array[:, 1]):.4f}")
    print(f"Melhor mês (RMSE): {min(monthly_metrics.items(), key=lambda x: x[1]['rmse'])[0]}")
    print(f"Pior mês (RMSE):   {max(monthly_metrics.items(), key=lambda x: x[1]['rmse'])[0]}")