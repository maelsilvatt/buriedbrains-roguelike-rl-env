# buriedbrains/agent_rules.py
import random
import copy
import math

# ==========================================
# CONSTANTES DE BALANCEAMENTO
# ==========================================

# Agente: Status Iniciais
INITIAL_HP = 300
INITIAL_BASE_DAMAGE = 10
INITIAL_EVASION = 0.05
INITIAL_ACCURACY = 0.90
INITIAL_CRIT_CHANCE = 0.05
INITIAL_CRIT_DAMAGE = 2.00

# Agente: Level Up Gains
LVL_UP_HP_GAIN = 10          # Vida ganha por nível
LVL_UP_DMG_GAIN = 2          # Dano base ganho por nível
LVL_UP_DEF_GAIN = 0.001      # Defesa ganha por nível (0.1%)

# --- CURVA DE XP (Progressão) ---
XP_EARLY_GAME_LIMIT = 10     # Até que nível é "Early Game"
XP_MID_GAME_LIMIT = 40       # Até que nível é "Mid Game"

# Early Game (Níveis 1-10)
XP_EARLY_BASE = 100
XP_EARLY_MULT = 25

# Mid Game (Níveis 11-40)
XP_MID_BASE = 400
XP_MID_MULT = 40

# Late Game (Nível 41+)
XP_LATE_BASE = 1600
XP_LATE_MULT = 100

# Matemática da Curva Logística
LOGISTIC_MIDPOINT_DEFAULT = 250
LOGISTIC_STEEPNESS_DEFAULT = 0.02

# Inimigos: Parâmetros de Nível
ENEMY_FLOORS_PER_LEVEL = 3        # A cada X andares, inimigo ganha +1 nível base
ENEMY_LEVEL_CAP_OFFSET = 4        # O inimigo pode ser (Andar + X) níveis
ENEMY_BOSS_LEVEL_EXTRA = 2        # Bosses podem exceder o cap em +X níveis

# Inimigos: Escalonamento de HP e Dano
# Parâmetros da curva logística para HP e Dano dos inimigos
ENEMY_SCALE_MIDPOINT = 200
ENEMY_SCALE_STEEPNESS = 0.015

# "Chaos Mode" (Late Game > 500 andares)
CHAOS_FLOOR_THRESHOLD = 500
CHAOS_GROWTH_FACTOR = 1.035       # Crescimento exponencial (1.035^excesso)

# Dano Fixo por Andar ("Floor Tax" para furar defesa)
ENEMY_FLOOR_FLAT_DMG_MULT = 0.8   # Multiplicador sobre o andar atual

# Multiplicadores de Tier (Elite/Boss batem mais forte)
ENEMY_ELITE_DMG_SCALE = 1.2
ENEMY_BOSS_DMG_SCALE = 1.5

# Status Base Genéricos de Inimigos
ENEMY_BASE_ACCURACY = 0.95
ENEMY_BASE_CRIT_CHANCE = 0.05
ENEMY_BASE_CRIT_DMG = 1.5
ENEMY_DEFAULT_BASE_DMG = 5

# Resistências a Status (Debuffs)
RESISTANCE_BOSS = 0.30            # 30% chance de ignorar status
RESISTANCE_ELITE = 0.15           # 15% chance de ignorar status

# Valores Default (Fallback se o YAML falhar)
DEFAULT_TIER_DATA = {
    'base_hp': 60, 
    'hp_variance': 0.1, 
    'target_hp_max': 30000, 
    'target_dmg_mult': 20.0
}
DEFAULT_ENEMY_XP = 30
DEFAULT_XP_GROWTH = 1.0

# Mapeamento de Nível Base por Tier
TIER_LEVEL_BASE = {
    'Fodder': 1, 
    'Common': 3, 
    'Professor': 5,
    'Tank': 7, 
    'Elite': 12, 
    'Boss': 20
}

# ==========================================
# LÓGICA DE PROGRESSÃO
# ==========================================

def create_initial_agent(name: str) -> dict:
    """
    Cria o estado inicial do agente no nível 1.
    """
    # Para o nível 1, usamos a lógica do Early Game
    initial_xp_req = XP_EARLY_BASE + (1 * XP_EARLY_MULT) 

    return {
        'name': name,
        'level': 1,
        'exp': 0,
        'exp_to_level_up': initial_xp_req,
        'hp': INITIAL_HP,
        'max_hp': INITIAL_HP,
        'base_stats': {
            'damage': INITIAL_BASE_DAMAGE,
            'defense': 0,
            'evasion': INITIAL_EVASION,
            'accuracy': INITIAL_ACCURACY,
            'crit_chance': INITIAL_CRIT_CHANCE,
            'crit_damage': INITIAL_CRIT_DAMAGE
        },
        'equipment': {
            'Weapon': 'Shortsword',
            'Armor': 'Leather Tunic',
            'Artifact': 'Amulet of Vigor'
        },
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
        'karma': {
            'real': 0.0,
            'imag': 0.0
        }
    }

def xp_to_next_level(level: int) -> int:
    """
    Curva 'Big Numbers': Projetada para levar o agente ao nível 50-60.
    """
    if level <= XP_EARLY_GAME_LIMIT:
        # Início Rápido
        return XP_EARLY_BASE + (level * XP_EARLY_MULT)
    
    elif level <= XP_MID_GAME_LIMIT:
        # Ritmo de Cruzeiro
        return XP_MID_BASE + (level - XP_EARLY_GAME_LIMIT) * XP_MID_MULT
        
    else:
        # Late Game
        return XP_LATE_BASE + (level - XP_MID_GAME_LIMIT) * XP_LATE_MULT

def check_for_level_up(agent_state: dict) -> bool:
    """
    Verifica e aplica subida de nível.
    """
    leveled_up = False
    
    while agent_state['exp'] >= agent_state['exp_to_level_up']:
        agent_state['exp'] -= agent_state['exp_to_level_up']
        agent_state['level'] += 1
        agent_state['exp_to_level_up'] = xp_to_next_level(agent_state['level'])

        # Aplicação dos ganhos de status definidos nas constantes
        agent_state['max_hp'] += LVL_UP_HP_GAIN
        agent_state['hp'] = agent_state['max_hp']

        agent_state['base_stats']['damage'] += LVL_UP_DMG_GAIN
        agent_state['base_stats']['defense'] += LVL_UP_DEF_GAIN
        
        leveled_up = True
        
    return leveled_up

def logistic_scale(
    floor: int,
    max_val: float,
    midpoint: float = LOGISTIC_MIDPOINT_DEFAULT,
    steepness: float = LOGISTIC_STEEPNESS_DEFAULT
) -> float:
    """
    Calcula um valor escalonado usando uma Curva Logística (Sigmoide).
    """
    # Valor logístico no andar atual
    val = max_val / (1 + math.exp(-steepness * (floor - midpoint)))

    # Valor logístico no andar 0 (offset inicial)
    offset = max_val / (1 + math.exp(-steepness * (0 - midpoint)))

    # Retornamos a curva deslocada para que comece em ~0
    return max(0, val - offset)

def calculate_enemy_xp(enemy_data: dict, floor: int, catalogs: dict) -> int:
    tier_name = enemy_data.get('tier', 'Common')
    tier_data = catalogs['enemy_tiers'].get(tier_name, {})
    
    base_xp = enemy_data.get('exp_yield', tier_data.get('xp_base', DEFAULT_ENEMY_XP))
    growth_per_floor = tier_data.get('xp_per_floor', DEFAULT_XP_GROWTH)
    
    total_xp = base_xp + (floor * growth_per_floor)

    return int(total_xp)

def instantiate_enemy(
    enemy_name: str, 
    agent_current_floor: int, 
    catalogs: dict, 
    room_effect_name: str = None
) -> dict:
    
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
        tier_data = DEFAULT_TIER_DATA

    # Cálculo do Nível do Inimigo
    calculated_level = (
        TIER_LEVEL_BASE.get(tier_name, 1)
        + (max(1, agent_current_floor) // ENEMY_FLOORS_PER_LEVEL)
    )

    max_level_allowed = agent_current_floor + ENEMY_LEVEL_CAP_OFFSET

    if tier_name == 'Boss':
        max_level_allowed += ENEMY_BOSS_LEVEL_EXTRA

    enemy_level = max(1, min(calculated_level, max_level_allowed))

    # Extração de dados do tier
    base_hp = tier_data.get('base_hp', DEFAULT_TIER_DATA['base_hp'])
    hp_variance = tier_data.get('hp_variance', DEFAULT_TIER_DATA['hp_variance'])
    target_hp_max = float(tier_data.get('target_hp_max', DEFAULT_TIER_DATA['target_hp_max']))
    target_dmg_mult = float(tier_data.get('target_dmg_mult', DEFAULT_TIER_DATA['target_dmg_mult']))

    floor = max(1, agent_current_floor)

    # Lógica de escalonamento de HP e Dano
    if floor <= CHAOS_FLOOR_THRESHOLD:
        # Crescimento Logístico
        hp_growth = logistic_scale(
            floor, 
            max_val=target_hp_max, 
            midpoint=ENEMY_SCALE_MIDPOINT, 
            steepness=ENEMY_SCALE_STEEPNESS
        )
        dmg_mult_growth = logistic_scale(
            floor, 
            max_val=target_dmg_mult, 
            midpoint=ENEMY_SCALE_MIDPOINT, 
            steepness=ENEMY_SCALE_STEEPNESS
        )

        final_hp = int(base_hp + hp_growth)
        final_dmg_mod = 1.0 + dmg_mult_growth
    else:
        # Late game (Chaos Mode)
        excess_floors = floor - CHAOS_FLOOR_THRESHOLD
        chaos_factor = CHAOS_GROWTH_FACTOR ** excess_floors
        final_hp = (base_hp + target_hp_max) * chaos_factor
        final_dmg_mod = (1.0 + target_dmg_mult) * chaos_factor
    
    # Dano Fixo por Andar (Floor Tax)
    floor_flat_damage = floor * ENEMY_FLOOR_FLAT_DMG_MULT

    # Ajuste por Tier
    tier_scaling_factor = 1.0
    if tier_name == 'Elite':
        tier_scaling_factor = ENEMY_ELITE_DMG_SCALE
    elif tier_name == 'Boss':
        tier_scaling_factor = ENEMY_BOSS_DMG_SCALE
        
    adjusted_floor_damage = floor_flat_damage * tier_scaling_factor

    # Dano Base do YAML escalado
    base_damage_from_yaml = enemy_data.get('base_stats', {}).get('damage', ENEMY_DEFAULT_BASE_DMG)
    scaled_base_damage = base_damage_from_yaml * final_dmg_mod

    final_damage_value = int(scaled_base_damage + adjusted_floor_damage)
    
    # Aplica variância no HP e Dano
    variance = random.uniform(1.0 - hp_variance, 1.0 + hp_variance)
    final_hp = int(final_hp * variance)
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
            'accuracy': ENEMY_BASE_ACCURACY,
            'crit_chance': ENEMY_BASE_CRIT_CHANCE,
            'crit_damage': ENEMY_BASE_CRIT_DMG,
            'status_resistance': 0.0,
            'damage': final_damage_value 
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
            if stat == 'damage':
                continue 
            enemy_instance['base_stats'][stat] = value

    # Ajustes de Resistência por Tier (Constants)
    if tier_name == 'Boss':
        enemy_instance['base_stats']['status_resistance'] = max(
            enemy_instance['base_stats'].get('status_resistance', 0), RESISTANCE_BOSS
        )
    elif tier_name in ['Elite', 'Tank']:
        enemy_instance['base_stats']['status_resistance'] = max(
            enemy_instance['base_stats'].get('status_resistance', 0), RESISTANCE_ELITE
        )

    # Efeitos de Sala
    if room_effect_name:
        room_rule = catalogs.get('room_effects', {}).get(room_effect_name)
        if room_rule and 'stat_modifier' in room_rule:
            for stat, value in room_rule['stat_modifier'].items():
                if stat in enemy_instance['base_stats']:
                    enemy_instance['base_stats'][stat] += value

    return enemy_instance