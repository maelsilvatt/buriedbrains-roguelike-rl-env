# buriedbrains/content_generation.py
import random
from typing import Dict

def _calculate_room_budget(floor: int, multiplier: float) -> float:
    """    
    Define o orçamento que o gerador tem para gastar baseado no andar.
    
    Fórmula: (Base + (Andar * Escala)) * Multiplicador
    """
    # Constantes Econômicas
    BASE_BUDGET = 100.0
    FLOOR_SCALING = 10.0
    
    # Cálculo bruto
    raw_budget = BASE_BUDGET + (floor * FLOOR_SCALING)
    
    # Aplica o multiplicador do ambiente (1.0 = Normal, 2.0 = Dobro de recursos/perigo)
    return raw_budget * multiplier

def generate_room_content(
    catalogs: Dict,
    current_floor: int,
    budget_multiplier: float = 1.0,
    guarantee_enemy: bool = False
) -> Dict:
    """
    Gera o conteúdo de uma sala baseado no andar atual, orçamentos e catálogos fornecidos. 
    Retorna um dicionário com inimigos, eventos, itens e efeitos de sala selecionados.
    """

    current_floor = int(current_floor)    
    budget = _calculate_room_budget(current_floor, budget_multiplier)
    selected_content = {'enemies': [], 'events': [], 'items': [], 'room_effects': []}
    
    processing_order = ['enemies', 'events', 'room_effects']
    current_budget = budget
    enemy_chosen_data = None 

    pool_map = {
        'enemies': catalogs.get('enemies', {}),
        'events': catalogs.get('events', {}),
        'room_effects': catalogs.get('room_effects', {})
    }
    equipment_catalog = catalogs.get('equipment', {}) 

    # Processa cada pool na ordem definida
    for pool_name in processing_order:
        current_pool = pool_map.get(pool_name, {})        
        all_candidates_values = list(current_pool.values())
        
        # Filtra apenas candidatos elegíveis
        candidates = []
        for c in all_candidates_values:
            if not isinstance(c, dict): 
                continue
            
            m_floor = int(c.get('min_floor', 0))            

            # Andar Atual >= Min Floor
            if current_floor < m_floor:
                continue # PULA ESSE CANDIDATO

            # Se passou por tudo, adiciona
            candidates.append(c)

        if not candidates:
            continue
        
        # Slot ci: Inimigos
        if pool_name == 'enemies':
            weights = [c.get('weight', 0) for c in candidates]
            if any(w > 0 for w in weights):
                try:
                    chosen = random.choices(candidates, weights=weights, k=1)[0]
                    cost = chosen.get('cost', 0)
                    if budget >= cost:
                        enemy_chosen_data = chosen
                except (IndexError, ValueError): pass

            # Fallback
            if enemy_chosen_data is None and guarantee_enemy:
                 eligible = [c for c in candidates if c.get('cost', float('inf')) > 0 and c.get('tier') != 'Boss']
                 if eligible:
                      enemy_chosen_data = min(eligible, key=lambda e: e.get('cost', float('inf')))

            if enemy_chosen_data:
                 selected_content['enemies'].append(enemy_chosen_data['name'])
                 current_budget -= enemy_chosen_data.get('cost', 0)

        # Slot ce: Eventos/Itens
        elif pool_name == 'events':
            dynamic_weights = []
            enemy_tier = enemy_chosen_data.get('tier', 'Common') if enemy_chosen_data else None
            is_high_tier = enemy_tier in ['Elite', 'Boss']
            
            for c in candidates:
                name = c.get('name')
                w = c.get('weight', 0)
                if name == 'Morbid Treasure':
                    dynamic_weights.append(w * 20 if is_high_tier else 0)
                elif name == 'Treasure':
                    dynamic_weights.append(w * 0.1 if is_high_tier else w)
                else:
                    dynamic_weights.append(w)

            if any(w > 0 for w in dynamic_weights):
                try:
                    ev = random.choices(candidates, weights=dynamic_weights, k=1)[0]
                    cost = ev.get('cost', 0)
                    name = ev.get('name') 
                    
                    if cost <= 0 or current_budget >= cost:
                        if name in ['Treasure', 'Morbid Treasure']:
                            selected_content['events'].append(name)
                            # Lógica de Loot Simplificada
                            is_tome = random.random() < (0.3 if name == 'Morbid Treasure' else 0.15)
                            target_rarity = ['Epic', 'Legendary'] if name == 'Morbid Treasure' else ['Common', 'Rare']
                            
                            if is_tome:
                                pool = [n for n, d in equipment_catalog.items() if d.get('type') == 'SkillTome' and d.get('rarity') in target_rarity]
                                if pool: selected_content['items'].append(random.choice(pool))
                            else:
                                pool = [n for n, d in equipment_catalog.items() if d.get('type') != 'SkillTome' and d.get('rarity') in target_rarity]
                                if pool: 
                                    qty = 1 + (1 if random.random() < 0.2 else 0)
                                    for _ in range(qty): selected_content['items'].append(random.choice(pool))
                        elif name != 'None':
                            selected_content['events'].append(name)
                        
                        if cost > 0: current_budget -= cost
                except (IndexError, ValueError): pass

        # Slot cf: Efeitos de Sala
        elif pool_name == 'room_effects':
            weights = [c.get('weight', 0) for c in candidates]
            if any(w > 0 for w in weights):
                try:
                    chosen = random.choices(candidates, weights=weights, k=1)[0]
                    if current_budget >= chosen.get('cost', 0):
                        selected_content['room_effects'].append(chosen['name'])
                        current_budget -= chosen.get('cost', 0)
                except (IndexError, ValueError): pass

    return selected_content