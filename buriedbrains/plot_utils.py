# buriedbrains/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def save_poincare_plot(karma_history: list, agent_name: str, save_path: str):
    """
    Gera e salva a trajetória de Karma no Disco de Poincaré.
    """
    if not karma_history:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Desenhar o Disco
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Desenhar os Polos (Saint vs Villain)
    # (padrão do reputation.py: 0.95 e -0.95)
    ax.scatter([0.95], [0], color='blue', marker='*', s=150, label='Santo', zorder=5)
    ax.scatter([-0.95], [0], color='red', marker='X', s=120, label='Vilão', zorder=5)
    
    # 3. Plotar a Trajetória
    # Converte a lista de complexos (ou dicts {'real':.., 'imag':..}) para arrays
    x = []
    y = []
    for k in karma_history:
        if isinstance(k, complex):
            x.append(k.real)
            y.append(k.imag)
        elif isinstance(k, dict):
            x.append(k.get('real', 0))
            y.append(k.get('imag', 0))
    
    # Desenha a linha do tempo da trajetória
    ax.plot(x, y, color='purple', linewidth=1.5, alpha=0.7, label='Trajetória')
    
    # Marca o Início e o Fim
    ax.scatter(x[0], y[0], color='white', edgecolor='black', s=60, marker='o', label='Início', zorder=6)
    ax.scatter(x[-1], y[-1], color='purple', edgecolor='black', s=80, marker='o', label='Fim', zorder=6)

    # Estilo
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Trajetória de Karma: {agent_name}", fontsize=12)
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
    ax.axis('off') # Remove eixos quadrados feios

    # Salvar
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) 