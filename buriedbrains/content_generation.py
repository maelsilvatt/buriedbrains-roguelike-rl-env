# buriedbrains/content_generation.py
import random
from typing import Dict

# ==========================================
# CONSTANTES DE GERAÇÃO DE CONTEÚDO
# ==========================================
# Economia de Sala
BASE_BUDGET = 100.0          # Orçamento base fixo
FLOOR_SCALING = 10.0         # Orçamento extra por andar

# Eventos Especiais de Loot
# Se o inimigo da sala for de tier alto (Elite/Boss):
WEIGHT_MULT_MORBID_HIGH = 20.0  # Chance de 'Morbid Treasure' aumenta 20x
WEIGHT_MULT_TREASURE_HIGH = 0.1 # Chance de 'Treasure' normal cai para 10%

# Definição do que é "Tier Alto" para triggering de eventos especiais
HIGH_TIER_ENEMIES = ['Elite', 'Boss']

# Propriedades dos baús
# Chance de vir um Grimório (SkillTome) ao abrir baú
CHANCE_TOME_MORBID = 0.30    # 30% em baús mórbidos
CHANCE_TOME_STANDARD = 0.15  # 15% em baús normais

# Raridades permitidas por tipo de baú
RARITY_POOL_MORBID = ['Epic', 'Legendary']
RARITY_POOL_STANDARD = ['Common', 'Rare']

# Chance de vir item extra (Double Drop)
CHANCE_EXTRA_ITEM = 0.20     # 20% de chance de dropar +1 item
BASE_ITEM_QTY = 1            # Quantidade base
EXTRA_ITEM_QTY = 1           # Quantidade adicional se proc

# Configuração da ordem de geração (c1, c2, c3)
GENERATION_ORDER = ['enemies', 'events', 'room_effects']
FALLBACK_ENEMY_TIER_BLACKLIST = ['Boss'] # Se falhar, nunca puxa Boss

# ==========================================
# LÓGICA DE GERAÇÃO
# ==========================================
def _calculate_room_budget(floor: int, multiplier: float) -> float:
    """    
    Define o orçamento que o gerador tem para gastar baseado no andar.
    Fórmula: (Base + (Andar * Escala)) * Multiplicador
    """
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
    
    current_budget = budget
    enemy_chosen_data = None 

    pool_map = {
        'enemies': catalogs.get('enemies', {}),
        'events': catalogs.get('events', {}),
        'room_effects': catalogs.get('room_effects', {})
    }
    equipment_catalog = catalogs.get('equipment', {}) 

    # Processa cada pool na ordem definida
    for pool_name in GENERATION_ORDER:
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
                continue # Fora

            # Se passou por tudo, adiciona
            candidates.append(c)

        if not candidates:
            continue
                
        # c1: Inimigos        
        if pool_name == 'enemies':
            weights = [c.get('weight', 0) for c in candidates]
            if any(w > 0 for w in weights):
                try:
                    chosen = random.choices(candidates, weights=weights, k=1)[0]
                    cost = chosen.get('cost', 0)
                    if budget >= cost:
                        enemy_chosen_data = chosen
                except (IndexError, ValueError): pass

            # Fallback: Se não conseguiu comprar nada e é obrigado a ter inimigo
            if enemy_chosen_data is None and guarantee_enemy:
                 eligible = [
                     c for c in candidates 
                     if c.get('cost', float('inf')) > 0 
                     and c.get('tier') not in FALLBACK_ENEMY_TIER_BLACKLIST
                 ]
                 if eligible:
                      # Pega o mais barato possível para não quebrar o jogo
                      enemy_chosen_data = min(eligible, key=lambda e: e.get('cost', float('inf')))

            if enemy_chosen_data:
                 selected_content['enemies'].append(enemy_chosen_data['name'])
                 current_budget -= enemy_chosen_data.get('cost', 0)
        
        # c2: Eventos e Itens (Loot)        
        elif pool_name == 'events':
            dynamic_weights = []
            
            # Verifica se o inimigo escolhido anteriormente é forte
            enemy_tier = enemy_chosen_data.get('tier', 'Common') if enemy_chosen_data else None
            is_high_tier = enemy_tier in HIGH_TIER_ENEMIES
            
            for c in candidates:
                name = c.get('name')
                w = c.get('weight', 0)
                
                # Ajuste dinâmico de pesos baseado no perigo da sala
                if name == 'Morbid Treasure':
                    # Se tem Boss/Elite, a chance de tesouro bom dispara
                    dynamic_weights.append(w * WEIGHT_MULT_MORBID_HIGH if is_high_tier else 0)
                elif name == 'Treasure':
                    # Se tem Boss/Elite, o tesouro comum fica raro (preferimos o Morbid)
                    dynamic_weights.append(w * WEIGHT_MULT_TREASURE_HIGH if is_high_tier else w)
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
                            
                            # Define se é baú mórbido
                            is_morbid = (name == 'Morbid Treasure')
                            
                            # Define chance de vir grimório (Tome)
                            tome_chance = CHANCE_TOME_MORBID if is_morbid else CHANCE_TOME_STANDARD
                            is_tome = random.random() < tome_chance
                            
                            # Define pool de raridade
                            target_rarity = RARITY_POOL_MORBID if is_morbid else RARITY_POOL_STANDARD
                            
                            if is_tome:
                                # Filtra Grimórios
                                pool = [
                                    n for n, d in equipment_catalog.items() 
                                    if d.get('type') == 'SkillTome' and d.get('rarity') in target_rarity
                                ]
                                if pool: 
                                    selected_content['items'].append(random.choice(pool))
                            else:
                                # Filtra Equipamentos Normais
                                pool = [
                                    n for n, d in equipment_catalog.items() 
                                    if d.get('type') != 'SkillTome' and d.get('rarity') in target_rarity
                                ]
                                if pool: 
                                    # Chance de vir mais de um item
                                    qty = BASE_ITEM_QTY + (EXTRA_ITEM_QTY if random.random() < CHANCE_EXTRA_ITEM else 0)
                                    for _ in range(qty): 
                                        selected_content['items'].append(random.choice(pool))
                        
                        elif name != 'None':
                            selected_content['events'].append(name)
                        
                        if cost > 0: 
                            current_budget -= cost
                except (IndexError, ValueError): pass
        
        # c3: Efeitos de Sala        
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