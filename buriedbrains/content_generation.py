# buriedbrains/content_generation.py
import random
from typing import Dict, List

def _calculate_costs(pools: Dict) -> Dict:
    """
    Calcula o 'custo' médio de cada pool (inimigos, itens, etc.)
    para normalizar as probabilidades de seleção durante a geração.
    """
    costs = {}
    for pool_name, items in pools.items():
        if not items:
            costs[pool_name] = 1
            continue
        
        total_value = sum(item.get('value', 0) for item in items.values())
        costs[pool_name] = total_value / len(items) if len(items) > 0 else 1
    return costs

def generate_room_content(
    pools: Dict, 
    costs: Dict, 
    budget: float, 
    current_floor: int,
    guarantee_enemy: bool = False  # >> MODIFICAÇÃO AQUI: Novo parâmetro opcional <<
) -> Dict:
    """
    Popula uma sala com conteúdo (inimigos, itens, efeitos) de forma
    balanceada, respeitando um orçamento de dificuldade (budget).
    """
    selected_content = {pool_name: [] for pool_name in pools.keys()}
    
    sorted_pools = sorted(
        pools.keys(), 
        key=lambda p: {'enemies': 0, 'items': 1, 'room_effects': 2}.get(p, 3)
    )

    for pool_name in sorted_pools:
        candidates = list(pools[pool_name].values())
        candidates = [c for c in candidates if c.get('min_floor', 0) <= current_floor]
        if not candidates:
            continue

        weights = [c.get('value', 0) * c.get('rarity_multiplier', 1) for c in candidates]
        
        while budget > 0 and any(w > 0 for w in weights):
            try:
                chosen_entity = random.choices(candidates, weights=weights, k=1)[0]
                entity_cost = costs.get(pool_name, 1)

                if budget >= entity_cost:
                    selected_content[pool_name].append(chosen_entity['name'])
                    budget -= entity_cost
                    idx = candidates.index(chosen_entity)
                    weights[idx] = 0
                else:
                    break
            except IndexError:
                break

        # >> MODIFICAÇÃO AQUI: Lógica para garantir um inimigo <<
        if guarantee_enemy and pool_name == 'enemies' and not selected_content['enemies']:
            # Se a flag estiver ativa, for a pool de inimigos e nenhum inimigo foi selecionado,
            # adicionamos o inimigo mais barato possível.
            if candidates:
                cheapest_enemy = min(candidates, key=lambda e: e.get('value', float('inf')))
                enemy_cost = costs.get('enemies', 1)
                
                # Adicionamos mesmo que o orçamento seja baixo, garantindo o inimigo para o teste.
                selected_content['enemies'].append(cheapest_enemy['name'])
                budget -= enemy_cost # Deduz o custo para afetar a geração de itens.
                
    return selected_content