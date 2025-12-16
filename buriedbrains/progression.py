# buriedbrains/agent_rules.py
import random
import copy
import math

# Constantes do agente
INITIAL_HP = 300
INITIAL_BASE_DAMAGE = 10
BASE_XP_REQ = 30
XP_REQ_INC = 4  # crescimento linear do xp necessário por nível

def create_initial_agent(name: str) -> dict:
    """
    Cria o estado inicial do agente no nível 1.
    """
    return {
        'name': name,
        'level': 1,
        'exp': 0,
        'exp_to_level_up': BASE_XP_REQ,
        'hp': INITIAL_HP,
        'max_hp': INITIAL_HP,
        'base_stats': {
            'damage': INITIAL_BASE_DAMAGE,
            'defense': 0,
            'evasion': 0.05,    # chance base de esquiva
            'accuracy': 0.90,   # chance base de acerto
            'crit_chance': 0.05,
            'crit_damage': 2.00 # padrão 2x dano ao dar crítico
        },
        'equipment': {
            'Weapon': 'Shortsword',       # equipamento inicial
            'Armor': 'Leather Tunic',     # equipamento inicial
            'Artifact': 'Amulet of Vigor' # equipamento inicial
        },
        # Deck inicial (slots 0-3)
        'skills': [
            'Quick Strike', 
            'Heavy Blow', 
            'Stone Shield', 
            'Wait'
        ],
        'persistent_effects': {},
        'effects': {},
        'cooldowns': {
            'Quick Strike': 0,
            'Heavy Blow': 0,
            'Stone Shield': 0,
            'Wait': 0
        },
        # Estado social (karma)
        'karma': {
            'real': 0.0,
            'imag': 0.0
        }
    }

def xp_to_next_level(level: int) -> int:
    """
    Curva 'Big Numbers': Projetada para levar o agente ao nível 50-60.
    Níveis iniciais são rápidos (2-3 lutas).
    Mid-game mantém ritmo constante (3-5 lutas).
    """
    if level <= 10:
        # Início Rápido: Nível 1 custa 100. Nível 10 custa ~300.
        # Agente upa quase a cada 2 salas.
        return 100 + (level * 25)
    
    elif level <= 40:
        # Ritmo de Cruzeiro: Nível 20 custa 600. Nível 40 custa 1200.
        # Com o XP dos inimigos escalando, mantém ~4 lutas por nível.
        return 400 + (level - 10) * 40
        
    else:
        # Late Game: Escala para evitar nível 200.
        return 1600 + (level - 40) * 100

def check_for_level_up(agent_state: dict) -> bool:
    """
    Verifica e aplica subida de nível.
    Suporta múltiplos níveis de uma vez.
    Retorna True se subiu pelo menos um nível.
    """
    leveled_up = False
    
    while agent_state['exp'] >= agent_state['exp_to_level_up']:
        agent_state['exp'] -= agent_state['exp_to_level_up']
        agent_state['level'] += 1
        agent_state['exp_to_level_up'] = xp_to_next_level(agent_state['level'])

        # Aumento de vida mais contido para forçar dependência de armadura
        agent_state['max_hp'] += 5 # era 10
        agent_state['hp'] = agent_state['max_hp']

        # Crescimento de dano propositalmente baixo para valorizar troca de arma
        agent_state['base_stats']['damage'] += 1

        # Defesa cresce devagar para evitar escalonamento infinito
        agent_state['base_stats']['defense'] += 0.001 # força o agente a buscar armaduras
        
        leveled_up = True
        
    return leveled_up

def logistic_scale(
    floor: int,
    max_val: float,
    midpoint: float = 250,
    steepness: float = 0.02
) -> float:
    """
    Calcula um valor escalonado usando uma Curva Logística (Sigmoide).
    
    A curva logística é ideal para progressões de RPG/RL porque:
    - Cresce devagar no início (evita power creep precoce).
    - Acelera fortemente no meio do jogo (onde o jogador/IA deve sentir evolução).
    - Desacelera no fim, aproximando-se suavemente de um teto (soft cap).

    Fórmula base utilizada:
        f(x) = L / (1 + e^(-k * (x - x0)))

        Onde:
            L   = valor máximo (max_val)
            k   = steepness (inclinação da curva)
            x0  = midpoint (ponto de inflexão)
            x   = floor (andar atual)

    Args:
        floor (int):
            O andar atual da run.
        max_val (float):
            Valor máximo que a curva tende a alcançar conforme o floor → infinito.
        midpoint (float):
            O andar onde a curva cresce mais rápido (ponto de inflexão).
        steepness (float):
            Controla o quão agressiva é a subida da curva.
            Valores maiores deixam a curva mais “vertical”.

    Returns:
        float: O valor escalonado correspondente ao andar atual,
               começando próximo de 0 e tendendo a max_val.
    """

    # Fórmula logística padrão:
    #   L / (1 + e^(-k * (x - x0)))
    #
    # Porém, se aplicada diretamente, no floor=0 o valor já seria > 0.
    # Para garantir que a curva comece praticamente em 0,
    # calculamos o valor no floor=0 e subtraímos como offset.

    # Valor logístico no andar atual
    val = max_val / (1 + math.exp(-steepness * (floor - midpoint)))

    # Valor logístico no andar 0 (offset inicial)
    offset = max_val / (1 + math.exp(-steepness * (0 - midpoint)))

    # Retornamos a curva deslocada para que comece em ~0
    return max(0, val - offset)

def calculate_enemy_xp(enemy_data: dict, floor: int, catalogs: dict) -> int:
    tier_name = enemy_data.get('tier', 'Common')
    tier_data = catalogs['enemy_tiers'].get(tier_name, {})
    
    # Se o inimigo tem um valor específico no YAML, usa ele.
    # Senão, usa a base genérica do Tier.
    base_xp = enemy_data.get('exp_yield', tier_data.get('xp_base', 30))
    
    # Pega o multiplicador do Tier (ex: Boss cresce 5.0 por andar)
    # Se não tiver definido, usa 1.0 como segurança
    growth_per_floor = tier_data.get('xp_per_floor', 1.0)
    
    # Fórmula: Base Específica + (Andar * Crescimento do Tier)
    total_xp = base_xp + (floor * growth_per_floor)

    return int(total_xp)

def instantiate_enemy(
    enemy_name: str, 
    agent_current_floor: int, 
    catalogs: dict, 
    room_effect_name: str = None
) -> dict:
    
    # Recupera dados do pool de inimigos
    enemy_pool = catalogs.get('enemies', {})
    enemy_data = enemy_pool.get(enemy_name)    
 
    if not enemy_data:        
        enemy_data = enemy_pool.get('Zombie', {})
        enemy_name = "Unknown Minion"

    # Carrega dados base
    tags = enemy_data.get('tags', [])
    base_skills = enemy_data.get('skills', ['Wait'])    
    tier_name = enemy_data.get('tier', 'Common')
    tier_data = catalogs.get('enemy_tiers', {}).get(tier_name)

    if not tier_data:
        tier_data = {
            'base_hp': 60, 'hp_variance': 0.1, 
            'target_hp_max': 30000, 'target_dmg_mult': 20.0
        }

    # Nível base por tier + progressão por andar
    TIER_LEVEL_BASE = {
        'Fodder': 1, 'Common': 3, 'Professor': 5,
        'Tank': 7, 'Elite': 12, 'Boss': 20
    }
    FLOORS_PER_LEVEL = 3

    # Nível calculado pelo tier + andar
    calculated_level = (
        TIER_LEVEL_BASE.get(tier_name, 1)
        + (max(1, agent_current_floor) // FLOORS_PER_LEVEL)
    )

    # Level cap baseado no andar (evita inimigos impossíveis)
    max_level_allowed = agent_current_floor + 4

    # Bosses podem ultrapassar um pouco o limite
    if tier_name == 'Boss':
        max_level_allowed += 2

    # Aplica o cap e garante mínimo de 1
    enemy_level = max(1, min(calculated_level, max_level_allowed))

    base_hp = tier_data['base_hp']
    hp_variance = tier_data['hp_variance']
    target_hp_max = float(tier_data['target_hp_max'])
    target_dmg_mult = float(tier_data['target_dmg_mult'])

    floor = max(1, agent_current_floor)

    # --- 1. LÓGICA DE ESCALONAMENTO (CURVAS) ---
    if floor <= 500:
        # Crescimento de HP e Multiplicador de Dano via Logística
        hp_growth = logistic_scale(floor, max_val=target_hp_max, midpoint=200, steepness=0.015)
        dmg_mult_growth = logistic_scale(floor, max_val=target_dmg_mult, midpoint=200, steepness=0.015)

        final_hp = int(base_hp + hp_growth)
        final_dmg_mod = 1.0 + dmg_mult_growth
    else:
        # Late game (Caos)
        excess_floors = floor - 500
        chaos_factor = 1.035 ** excess_floors
        final_hp = (base_hp + target_hp_max) * chaos_factor
        final_dmg_mod = (1.0 + target_dmg_mult) * chaos_factor
    
    # Dano Fixo por Andar (Floor Tax) para perfurar armadura    
    floor_flat_damage = floor * 1.2

    # Ajuste por Tier (Bosses aproveitam mais o dano do andar)
    tier_scaling_factor = 1.0
    if tier_name == 'Elite':
        tier_scaling_factor = 1.2
    elif tier_name == 'Boss':
        tier_scaling_factor = 1.5
        
    adjusted_floor_damage = floor_flat_damage * tier_scaling_factor

    # Dano Base do YAML escalado pelo Multiplicador Logístico
    base_damage_from_yaml = enemy_data.get('base_stats', {}).get('damage', 5)
    scaled_base_damage = base_damage_from_yaml * final_dmg_mod

    # Evita multiplicação explosiva e garante dano mínimo no mid-game
    final_damage_value = int(scaled_base_damage + adjusted_floor_damage)
    
    # Aplica variância no HP
    variance = random.uniform(1.0 - hp_variance, 1.0 + hp_variance)
    final_hp = int(final_hp * variance)
    
    # Aplica variância no Dano (opcional, para não ser robótico)
    final_damage_value = int(final_damage_value * variance)

    # Calcula XP
    final_exp = calculate_enemy_xp(enemy_data, floor, catalogs)

    # Processa Skills
    skills_data = {}
    for skill_name in base_skills:
        original_skill_data = catalogs.get('skills', {}).get(skill_name)
        if original_skill_data:
            skills_data[skill_name] = copy.deepcopy(original_skill_data)

    # Monta Instância
    enemy_instance = {
        'name': enemy_name,
        'level': enemy_level,
        'max_hp': final_hp,
        'hp': final_hp,
        'effects': {},
        'base_stats': {
            'defense': 0.0,
            'evasion': 0.0,
            'accuracy': 0.95,
            'crit_chance': 0.05,
            'crit_damage': 1.5,
            'status_resistance': 0.0,
            # Placeholder, será preenchido abaixo
            'damage': 0 
        },
        'skills': skills_data,
        'cooldowns': {skill: 0 for skill in base_skills},
        'tags': tags,
        'tier': tier_name,
        'exp_yield': final_exp,
        'team': 2
    }

    # Copia outros stats base do YAML, se houver
    if 'base_stats' in enemy_data:
        for stat, value in enemy_data['base_stats'].items():            
            
            # Protege o dano calculado
            if stat == 'damage':
                continue 
            
            enemy_instance['base_stats'][stat] = value

    # Injeta o valor calculado de dano
    enemy_instance['base_stats']['damage'] = final_damage_value 

    # Ajustes de Resistência por Tier
    if tier_name == 'Boss':
        enemy_instance['base_stats']['status_resistance'] = max(
            enemy_instance['base_stats'].get('status_resistance', 0), 0.3
        )
    elif tier_name in ['Elite', 'Tank']:
        enemy_instance['base_stats']['status_resistance'] = max(
            enemy_instance['base_stats'].get('status_resistance', 0), 0.15
        )

    # Efeitos de Sala
    if room_effect_name:
        room_rule = catalogs.get('room_effects', {}).get(room_effect_name)
        if room_rule and 'stat_modifier' in room_rule:
            for stat, value in room_rule['stat_modifier'].items():
                if stat in enemy_instance['base_stats']:
                    enemy_instance['base_stats'][stat] += value

    return enemy_instance
