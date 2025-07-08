import random
import os
import time

# --- CONTROLES AJUSTÁVEIS ---
# Altere estes valores para simular diferentes cenários
ANDAR_ATUAL_K = 4

# --- PARÂMETROS TOPOLÓGICOS DA SALA ATUAL ---
# Estes parâmetros seriam fornecidos pelo seu ambiente de simulação em tempo real
NUM_SALAS_VISITADAS = 8   # O agente já visitou 8 salas
TOTAL_SALAS_ANDAR = 10  # O andar atual tem 10 salas no total
bool_corredor = False       # Esta sala NÃO é um corredor (mude para True para testar)
# ---------------------------------------------


# --- CONFIGURAÇÃO DO AMBIENTE ---
# O "preço" de cada entidade em pontos de dificuldade
CONTENT_COSTS = {
    'Inimigo Comum': 10,
    'Inimigo Forte (H)': 25,
    'Inimigo de Elite (X)': 60,
    'Armadilha de Dano': 8,
    'Armadilha de Status': 12,
    'Tesouro Protegido (T)': -40, # Custo negativo = recompensa que "paga" por mais desafios
    'Fonte de Cura': -25,       # Custo negativo = recompensa
    'Efeito de Sala: Terreno Lento': 5,
    'Efeito de Sala: Zona de Evasão': -10
}

# Pools de conteúdo com pesos de seleção DENTRO de cada categoria
CONTENT_POOLS = {
    'INIMIGOS': {
        'Inimigo Comum': 70,
        'Inimigo Forte (H)': 25,
        'Inimigo de Elite (X)': 5, # Raro
    },
    'EVENTOS': {
        'Armadilha de Dano': 40,
        'Armadilha de Status': 25,
        'Tesouro Protegido (T)': 15, # Raro
        'Fonte de Cura': 20
    },
    'EFEITOS_SALA': {
        'Efeito de Sala: Terreno Lento': 60,
        'Efeito de Sala: Zona de Evasão': 40,
    }
}

# Probabilidades BASE de inclusão para cada categoria
BASE_CATEGORY_ADD_CHANCE = {
    'INIMIGOS': 0.80,      # 80% de chance base de ter um inimigo
    'EVENTOS': 0.50,       # 50% de chance base de ter um evento
    'EFEITOS_SALA': 0.35,  # 35% de chance base de ter um efeito
}

# --- FUNÇÕES AUXILIARES ---

def clear_screen():
    """Limpa o console para visualização em tempo real."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_generation_status(params, budget, alpha_v, message=""):
    """Exibe o estado atual da geração da sala, incluindo a topologia."""
    clear_screen()
    print("--- GERADOR DE CONTEÚDO GAMMA | BuriedBrains ---")
    print(f"Andar: {params['k']} | Progresso: {params['visitadas']}/{params['total']} | bool_corredor: {params['bool_corredor']}")
    print("-" * 60)
    print(f"ORÇAMENTO DE DIFICULDADE ATUAL: {budget:.2f}")
    print("-" * 60)
    print("Vetor alpha(v) em Geração:")
    if not any(alpha_v.values()):
        print("  (Vazio)")
    else:
        for category, items in alpha_v.items():
            if items:
                print(f"  - {category}: {items[0]}")
    print("-" * 60)
    if message:
        print(f"PASSO ATUAL: {message}\n")
    time.sleep(1.2)


# --- FUNÇÃO GAMMA (Versão Final com Topologia) ---

def gamma(floor_k, num_salas_visitadas, total_salas_andar, bool_corredor):
    """
    Função Gamma: Gera o conteúdo de uma sala (alpha_v) de forma procedural,
    influenciada pela topologia do grafo.
    """
    # 1. Calcular Orçamento com base na Topologia
    base_budget = 15 * (floor_k ** 1.2)
    
    progresso = num_salas_visitadas / total_salas_andar if total_salas_andar > 0 else 0
    bonus_progressao = 0.25 * progresso # Bônus de até 25% no final do andar
    
    fator_bool_corredor = 0.6 if bool_corredor else 1.0 # Penalidade de 40% para bool_corredores
    
    final_budget = (base_budget * (1 + bonus_progressao)) * fator_bool_corredor
    
    # 2. Ajustar Probabilidades de Categoria com base na Topologia
    adj_category_chance = BASE_CATEGORY_ADD_CHANCE.copy() # Copia as chances base
    
    if bool_corredor:
        adj_category_chance['EVENTOS'] = 0.10 # Chance muito baixa de eventos em bool_corredores
        adj_category_chance['EFEITOS_SALA'] = 0.15
        
    if progresso > 0.8: # Se estiver nos últimos 20% do andar
        adj_category_chance['INIMIGOS'] = 0.95 # Quase certeza de um inimigo final
        adj_category_chance['EVENTOS'] = 0.75  # Chance alta de um evento final (tesouro/armadilha)

    # Coleta de parâmetros para visualização
    params = {'k': floor_k, 'visitadas': num_salas_visitadas, 'total': total_salas_andar, 'bool_corredor': bool_corredor}
    alpha_v = {'INIMIGOS': [], 'EVENTOS': [], 'EFEITOS_SALA': []}
    display_generation_status(params, final_budget, alpha_v, "Iniciando geração com topologia...")

    # 3. Loop de Geração (itera sobre as categorias para preencher os "slots")
    for category in ['INIMIGOS', 'EVENTOS', 'EFEITOS_SALA']:
        if random.random() < adj_category_chance[category]:
            # Filtra o pool da categoria por itens que cabem no orçamento
            affordable_items = {
                item: weight for item, weight in CONTENT_POOLS[category].items()
                if CONTENT_COSTS[item] <= final_budget
            }
            
            if affordable_items:
                # Escolhe um item do pool FILTRADO e acessível
                chosen_item = random.choices(
                    population=list(affordable_items.keys()),
                    weights=list(affordable_items.values()),
                    k=1
                )[0]
                
                cost = CONTENT_COSTS[chosen_item]
                final_budget -= cost
                alpha_v[category].append(chosen_item)
                message = f"Categoria '{category}': Item '{chosen_item}' adicionado."
            else:
                message = f"Categoria '{category}': Nenhum item acessível com o orçamento atual."
        
        else:
            message = f"Categoria '{category}': Ignorada por sorteio de probabilidade."
            
        display_generation_status(params, final_budget, alpha_v, message)
        
    display_generation_status(params, final_budget, alpha_v, "Geração Finalizada!")
    return alpha_v

# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    
    # Chama a função Gamma com os parâmetros ajustáveis
    final_alpha_v = gamma(ANDAR_ATUAL_K, NUM_SALAS_VISITADAS, TOTAL_SALAS_ANDAR, bool_corredor)

    # Exibe um sumário final e limpo
    print("\n--- RESULTADO FINAL ---")
    print(f"Config: Andar {ANDAR_ATUAL_K}, Progresso {NUM_SALAS_VISITADAS}/{TOTAL_SALAS_ANDAR}, bool_corredor: {bool_corredor}")
    print("Conteúdo final (alpha_v) gerado:")
    if not any(final_alpha_v.values()):
        print("  A sala foi gerada vazia.")
    else:
        for category, items in final_alpha_v.items():
            if items:
                print(f"  - {category}: {items[0]}")
    print("-" * 25)