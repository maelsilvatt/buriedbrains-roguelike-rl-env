# buriedbrains/content_generation.py
import random
from typing import Dict, List

def _calculate_costs(pools: Dict) -> Dict:
    """
    Calcula o 'custo' médio de cada pool.
    """
    costs = {}
    for pool_name, items in pools.items():
        if not items:
            costs[pool_name] = 1
            continue

        # Usa 'cost' do YAML
        total_cost = sum(item.get('cost', 0) for item in items.values() if isinstance(item, dict))
        # Evita divisão por zero se a lista de items for vazia após filtragem
        num_items = len([item for item in items.values() if isinstance(item, dict)])
        costs[pool_name] = total_cost / num_items if num_items > 0 else 1
    return costs

# --- VERSÃO REFEITA ---
def generate_room_content(
    catalogs: Dict, # Modificado: Recebe todos os catálogos
    budget: float,
    current_floor: int,
    guarantee_enemy: bool = False
) -> Dict:
    """
    Popula uma sala seguindo a arquitetura Gamma (PDF) e regras de Tesouro:
    1. Seleciona no máximo 1 inimigo (ci).
    2. Seleciona no máximo 1 resultado de evento (ce):
        - Se Tesouro/Tesouro Mórbido: Gera 1 item de equipamento correspondente.
        - Se outro evento (não 'None'): Adiciona o nome do evento.
        - Se 'None': Nada acontece.
    3. Seleciona no máximo 1 efeito de sala (cf).
    """
    # Adicionada a chave 'items' para guardar equipamentos encontrados
    selected_content = {'enemies': [], 'events': [], 'items': [], 'room_effects': []}

    # Ordem definida pela arquitetura Gamma [cite: 555-560]
    processing_order = ['enemies', 'events', 'room_effects']
    current_budget = budget

    # Acesso aos catálogos individuais (assume que estão no nível superior de 'catalogs')
    enemy_pool = catalogs.get('enemies', {})
    event_pool = catalogs.get('events', {}) # Precisa garantir que 'events' esteja aqui
    effect_pool = catalogs.get('room_effects', {})
    equipment_catalog = catalogs.get('equipment', {}) # Necessário para gerar itens

    pool_map = {
        'enemies': enemy_pool,
        'events': event_pool,
        'room_effects': effect_pool
    }

    for pool_name in processing_order:

        current_pool = pool_map.get(pool_name, {})
        # Garante que estamos pegando os 'values' (os dicionários de dados)
        all_candidates_in_pool = list(current_pool.values())
        candidates = [
            c for c in all_candidates_in_pool
            if isinstance(c, dict) and c.get('min_floor', 0) <= current_floor
        ]

        if not candidates:
            continue # Pula se não houver candidatos válidos para o andar

        weights = [c.get('weight', 0) for c in candidates]

        # --- Slot ci: Inimigos (Máximo 1) ---
        if pool_name == 'enemies':
            enemy_chosen_data = None # Guarda o dicionário do inimigo escolhido
            if any(w > 0 for w in weights):
                try:
                    chosen_enemy_candidate = random.choices(candidates, weights=weights, k=1)[0]
                    enemy_cost = chosen_enemy_candidate.get('cost', 0)
                    # Adiciona se couber no budget inicial
                    if budget >= enemy_cost:
                        enemy_chosen_data = chosen_enemy_candidate
                except (IndexError, ValueError): pass # Ignora erros de seleção

            # Fallback da Garantia
            if enemy_chosen_data is None and guarantee_enemy:
                 eligible_enemies = [c for c in candidates if c.get('cost', float('inf')) > 0]
                 if eligible_enemies:
                      cheapest_enemy = min(eligible_enemies, key=lambda e: e.get('cost', float('inf')))
                      enemy_chosen_data = cheapest_enemy # Adiciona mesmo que estoure o budget

            if enemy_chosen_data:
                 selected_content['enemies'].append(enemy_chosen_data['name'])
                 # Sempre deduz o custo para afetar slots subsequentes
                 current_budget -= enemy_chosen_data.get('cost', 0) # Atualiza B_res1

        # --- Slot ce: Eventos/Itens (Máximo 1 Resultado) ---
        elif pool_name == 'events':
            if any(w > 0 for w in weights):
                try:
                    # Seleciona UM evento/resultado potencial baseado no peso
                    chosen_event = random.choices(candidates, weights=weights, k=1)[0]
                    event_cost = chosen_event.get('cost', 0)
                    event_name = chosen_event.get('name') # Assume 'name' foi injetado no __init__

                    # Processa o evento escolhido SOMENTE se couber no budget restante (B_res1)
                    # OU se o custo for não-positivo (recompensa/neutro)
                    if event_cost <= 0 or current_budget >= event_cost:
                        if event_name == 'Treasure':
                            # Filtra equipamentos Comuns ou Raros
                            eligible_items = [
                                name for name, data in equipment_catalog.items()
                                if isinstance(data, dict) and data.get('rarity') in ['Common', 'Rare']
                            ]
                            
                            if eligible_items:
                                # --- MUDANÇA: CHANCE DE MULTI-LOOT ---
                                # Garante 1 item, mas pode gerar até 3
                                num_items_to_drop = 1
                                
                                # 20% de chance de ter um segundo item
                                if random.random() < 0.20: 
                                    num_items_to_drop += 1
                                    # Se tiver o segundo, 10% de chance de ter um terceiro (total 2% chance)
                                    if random.random() < 0.10: 
                                        num_items_to_drop += 1
                                
                                for _ in range(num_items_to_drop):
                                    found_item = random.choice(eligible_items)
                                    selected_content['items'].append(found_item)                                

                            # Deduz o custo (apenas uma vez, para não punir o budget demais)
                            current_budget -= event_cost

                        elif event_name == 'Morbid Treasure':
                            # Filtra equipamentos Épicos ou Lendários
                            eligible_items = [
                                name for name, data in equipment_catalog.items()
                                if isinstance(data, dict) and data.get('rarity') in ['Epic', 'Legendary']
                            ]
                            if eligible_items:
                                found_item = random.choice(eligible_items)
                                selected_content['items'].append(found_item)
                            # Deduz o custo do EVENTO Tesouro Mórbido (negativo)
                            current_budget -= event_cost # Atualiza B_res2
                            # Lembre-se: A *causa* de Morbid Treasure (Boss/Elite) é externa a esta função.

                        elif event_name != 'None':
                            # Para outros eventos (Trap, Fountain, etc.)
                            selected_content['events'].append(event_name)
                            # Deduz o custo apenas se for positivo
                            if event_cost > 0:
                                 current_budget -= event_cost # Atualiza B_res2

                        # Se for 'None', não faz nada (custo 0)

                except (IndexError, ValueError):
                    pass # Ignora erros de seleção

        # --- Slot cf: Efeitos de Sala (Máximo 1) ---
        elif pool_name == 'room_effects':
            if any(w > 0 for w in weights):
                try:
                    chosen_effect_candidate = random.choices(candidates, weights=weights, k=1)[0]
                    effect_cost = chosen_effect_candidate.get('cost', 0)
                    # Só adiciona se couber no budget restante (B_res2)
                    if current_budget >= effect_cost:
                        selected_content['room_effects'].append(chosen_effect_candidate['name'])
                        current_budget -= effect_cost # Budget final
                except (IndexError, ValueError): pass

    return selected_content