# buriedbrains/plot_utils.py
import matplotlib

# Força o backend 'Agg' (Anti-Grain Geometry).
# Isso impede que o Matplotlib tente usar janelas (Tkinter),
# evitando o erro "main thread is not in main loop".
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import os

def save_poincare_plot(karma_history: list, agent_name: str, save_path: str):
    """
    Gera e salva a trajetória de Karma no Disco de Poincaré.
    """
    if not karma_history:
        return

    # Cria a figura (sem exibi-la)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Desenhar o Disco (Círculo Unitário)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Desenhar os Polos (Saint vs Villain)
    ax.scatter([0.95], [0], color='blue', marker='*', s=150, label='Santo', zorder=5)
    ax.scatter([-0.95], [0], color='red', marker='X', s=120, label='Vilão', zorder=5)
    
    # 3. Plotar a Trajetória
    x = []
    y = []
    for k in karma_history:
        if isinstance(k, complex):
            x.append(k.real)
            y.append(k.imag)
        elif isinstance(k, dict):
            x.append(k.get('real', 0))
            y.append(k.get('imag', 0))
    
    # Desenha a linha do tempo
    ax.plot(x, y, color='purple', linewidth=1.5, alpha=0.7, label='Trajetória')
    
    # Marca o Início e o Fim
    if x and y:
        ax.scatter(x[0], y[0], color='white', edgecolor='black', s=60, marker='o', label='Início', zorder=6)
        ax.scatter(x[-1], y[-1], color='purple', edgecolor='black', s=80, marker='o', label='Fim', zorder=6)

    # Estilização
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Trajetória de Karma: {agent_name}", fontsize=12)
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
    ax.axis('off')

    # Salvar e Fechar
    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    except Exception as e:
        print(f"[Plot Error] Falha ao salvar imagem: {e}")
    finally:
        # Fecha a figura explicitamente para liberar memória e evitar o erro do Tkinter
        plt.close(fig)