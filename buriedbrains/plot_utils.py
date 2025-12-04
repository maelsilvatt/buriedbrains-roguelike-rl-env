# buriedbrains/plot_utils.py
import matplotlib
# Força o backend 'Agg' para evitar erros de GUI
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def save_poincare_plot(karma_history: list, agent_name: str, save_path: str):
    """
    Gera e salva a trajetória de Karma no Disco de Poincaré
    """
    if not karma_history:
        return

    # Extração de dados
    x = []
    y = []
    for k in karma_history:
        if isinstance(k, complex):
            x.append(k.real)
            y.append(k.imag)
        elif isinstance(k, dict):
            x.append(k.get('real', 0))
            y.append(k.get('imag', 0))

    if not x or not y:
        return
    
    # Fundo cinza
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    
    # Desenhar o Disco e a Grid (Estilo Radar/Poincaré)
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Borda do limite (Círculo Unitário) - Linha grossa preta
    ax.plot(np.cos(theta), np.sin(theta), color='#333333', linewidth=2, zorder=1)
    # Preenchimento suave do disco
    ax.fill(np.cos(theta), np.sin(theta), color='#f0f0f5', alpha=0.3, zorder=0)

    # Círculos Concêntricos (Níveis de intensidade: 25%, 50%, 75%)
    for r in [0.25, 0.5, 0.75]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)

    # Eixos Cruzados (Crux)
    ax.plot([-1, 1], [0, 0], color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1) # Eixo Real
    ax.plot([0, 0], [-1, 1], color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1) # Eixo Imag

    # Anotações dos Polos    
    style_text = dict(fontsize=9, fontweight='bold', ha='center', va='center')
    
    # Santo (+Real)
    ax.text(1.15, 0, "SANTO", color='darkblue', **style_text)
    # Vilão (-Real)
    ax.text(-1.15, 0, "VILÃO", color='darkred', **style_text)
    # Caos/Ordem (Eixo Imaginário)s
    ax.text(0, 1.08, "+IMAG", color='gray', fontsize=7, ha='center')
    ax.text(0, -1.08, "-IMAG", color='gray', fontsize=7, ha='center')

    # Trajetória com Gradiente de Cor (Tempo) 
    # Cria segmentos de linha para aplicar gradiente (Cool -> Hot)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Cria um colormap personalizado: Azul (Início) -> Roxo -> Vermelho (Fim)
    cmap = LinearSegmentedColormap.from_list("time_gradient", ["#42a5f5", "#7e57c2", "#ef5350"])
    
    # Normaliza o tempo (0 a 1)
    norm = plt.Normalize(0, len(x))
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.8, zorder=3)
    lc.set_array(np.arange(len(x))) # Define a progressão para o colormap
    ax.add_collection(lc)

    # Início (Ponto menor, cor fria)
    ax.scatter(x[0], y[0], color='#42a5f5', edgecolor='black', s=50, marker='o', label='Início', zorder=4)
    # Fim (Ponto maior, estrela, cor quente)
    ax.scatter(x[-1], y[-1], color='#ef5350', edgecolor='black', s=100, marker='*', label='Fim', zorder=5)

    # Estilização
    ax.set_aspect('equal')
    # Margem extra para caber os textos fora do círculo
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    
    # Título 
    ax.set_title(f"Trajetória de Karma\nAgente: {agent_name}", fontsize=11, pad=15, color='#333333')
    
    # Remove a caixa quadrada padrão do matplotlib
    ax.axis('off')

    # Legenda
    ax.legend(loc='lower right', fontsize='x-small', framealpha=0.8, edgecolor='gray')

    # Salvar e Fechar
    try:        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    except Exception as e:
        print(f"[Plot Error] Falha ao salvar imagem: {e}")
    finally:
        plt.close(fig)