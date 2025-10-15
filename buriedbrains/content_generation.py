# buriedbrains/content_generation.py
import random
from typing import Dict, List

def _calculate_costs(pools: Dict) -> Dict:
    """
    Calcula o 'custo' médio de cada pool (inimigos, itens, etc.)
    para normalizar as probabilidades de seleção durante a geração.
    Esta é uma função auxiliar para a generate_room_content.
   
    """
    costs = {}
    for pool_name, items in pools.items():
        if not items:
            costs[pool_name] = 1  # Custo padrão para evitar divisão por zero
            continue
        
        # O valor total é a soma do 'value' de cada item na pool
        total_value = sum(item.get('value', 0) for item in items.values())
        costs[pool_name] = total_value / len(items) if len(items) > 0 else 1
    return costs

def generate_room_content(
    pools: Dict, 
    costs: Dict, 
    budget: float, 
    current_floor: int
) -> Dict:
    """
    Popula uma sala com conteúdo (inimigos, itens, efeitos) de forma
    balanceada, respeitando um orçamento de dificuldade (budget).
    Esta é a sua Função Gamma.
   
    """
    selected_content = {pool_name: [] for pool_name in pools.keys()}
    
    # Ordena as pools para seguir a lógica de dependência que você estabeleceu:
    # inimigos -> itens -> efeitos de sala
    sorted_pools = sorted(
        pools.keys(), 
        key=lambda p: {'enemies': 0, 'items': 1, 'room_effects': 2}.get(p, 3)
    )

    for pool_name in sorted_pools:
        # Pega a lista de candidatos (entidades) da pool atual
        candidates = list(pools[pool_name].values())
        
        # Filtra os candidatos pelo andar mínimo requerido, se aplicável
        candidates = [c for c in candidates if c.get('min_floor', 0) <= current_floor]
        if not candidates:
            continue

        # Calcula os pesos com base no valor e raridade de cada candidato
        weights = [c.get('value', 0) * c.get('rarity_multiplier', 1) for c in candidates]
        
        # Loop principal: tenta gastar o 'budget' com as entidades da pool atual
        while budget > 0 and any(w > 0 for w in weights):
            try:
                # Escolhe uma entidade com base nos pesos calculados
                chosen_entity = random.choices(candidates, weights=weights, k=1)[0]
                entity_cost = costs.get(pool_name, 1) # Usa o custo médio da pool

                if budget >= entity_cost:
                    selected_content[pool_name].append(chosen_entity['name'])
                    budget -= entity_cost
                    
                    # Impede que a mesma entidade seja escolhida novamente na mesma sala
                    idx = candidates.index(chosen_entity)
                    weights[idx] = 0
                else:
                    # Se não há orçamento nem para a entidade de menor custo, para a seleção desta pool
                    break
            except IndexError:
                # Ocorre se a lista de candidatos ou pesos ficar vazia
                break
                
    return selected_content