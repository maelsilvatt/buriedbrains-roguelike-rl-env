# buriedbrains/combat.py
import copy
import random
from typing import Dict, Any, List, Optional, Tuple

# ==========================================
# CONSTANTES DE COMBATE (CORE MECHANICS)
# ==========================================

# Caps de Stats
# Impede que agentes/inimigos fiquem imortais ou intocáveis
CAP_MIN_ACCURACY = 0.10       # Mínimo de 10% de chance de acerto (sempre há chance)
CAP_MAX_EVASION = 0.75        # Máximo de 75% de esquiva
CAP_MAX_CRIT = 1.00           # Máximo de 100% de crítico
CAP_MAX_DMG_REDUCTION = 0.70  # Máximo de 70% de redução de dano (Armor)

# Regras de Cálculo de Dano
HIT_CHANCE_FLOOR = 0.05       # Chance mínima absoluta de acertar (Roll puro)
HIT_CHANCE_CEILING = 0.95     # Chance máxima absoluta de acertar (Roll puro)
CRIT_MULTIPLIER_DEFAULT = 2.0 # Dano Crítico padrão (2x)
MIN_DEFENSE_MULTIPLIER = 0.1  # O alvo sempre toma pelo menos 10% do dano bruto

# Regras de Efeitos 
DEFAULT_EFFECT_DURATION = 2   # Duração padrão se não especificado
DEFAULT_REVIVE_POTENCY = 0.5  # Renasce com 50% da vida
ROOM_HEAL_DEBUFF_MOD = 0.5    # Penalidade de cura em salas profanas (50%)

# Recompensas de Consumíveis
REW_ITEM_FAIL = -5.0           # Tentou usar item inexistente (slot vazio)
REW_ITEM_WASTE = -2.0          # Usou item sem necessidade (vida cheia)
REW_ITEM_HEAL_BASE = 10.0      # Base para cura simples
REW_ITEM_HEAL_SCALAR = 0.1     # Multiplicador por ponto de HP curado
REW_ITEM_FULL_HEAL_BASE = 20.0 # Base para cura completa (Elixir)
REW_ITEM_BUFF_BASE = 5.0       # Recompensa por buffs de stat (Força/Defesa)
REW_ITEM_GOD_MODE = 50.0       # Recompensa por itens lendários (Ambrosia)

# ==========================================
# LÓGICA DE COMBATE
# ==========================================
def initialize_combatant(
    name: str,
    hp: int,
    equipment: list,
    skills: list,
    team: int,
    catalogs: Dict[str, Any],
    level: int = 1, 
    room_effect_name: Optional[str] = None,   
    saved_effects: Optional[Dict[str, Any]] = None, 
    base_damage: float = 0.0,
    base_defense: float = 0.0,
    base_evasion: float = 0.0,
    base_accuracy: float = 1.0,
    base_crit: float = 0.0
) -> Dict[str, Any]:
    """
    Prepara a 'ficha de personagem'.
    Aplica passivas de equipamentos e efeitos iniciais de sala.
    """
    skill_catalog = catalogs.get('skills', {})
    equipment_catalog = catalogs.get('equipment', {})
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})
        
    combatant = {
        'name': name,
        'hp': float(hp), 
        'max_hp': float(hp),
        'team': team,
        'level': level, 
        'skills': {s_name: skill_catalog.get(s_name, {}).copy() for s_name in skills if s_name in skill_catalog},
        'equipment': {e_name: equipment_catalog.get(e_name, {}).copy() for e_name in equipment if e_name in equipment_catalog},
        'cooldowns': {s: 0 for s in skills},
        'effects': {},

        'base_stats': {            
            'damage_modifier': 1.0,                        
            'flat_damage_bonus': float(base_damage),                     
            'damage_reduction': float(base_defense),                    
            'crit_chance': float(base_crit),
            'accuracy': float(base_accuracy) if base_accuracy > 0 else 1.0,                         
            'evasion_chance': float(base_evasion),        
            'dot_potency_modifier': 1.0,            
            'speed': 1.0, 
            'defense_taken': 1.0, 
            'magic_damage_modifier': 1.0,
            'physical_damage_modifier': 1.0
        },
        'on_hit_effects': [], 
        'on_being_hit_effects': [], 
        'special_effects': {} 
    }

    # Aplica passivas de Equipamentos
    for item_name, item_info in combatant['equipment'].items():
        if not isinstance(item_info, dict): continue 
        for effect, value in item_info.get('passive_effects', {}).items():
            if effect == 'flat_hp_bonus':
                combatant['hp'] += value
                combatant['max_hp'] += value
            elif effect in combatant['base_stats']:
                 combatant['base_stats'][effect] += value            

        if 'on_hit_effect' in item_info:
            combatant['on_hit_effects'].append(item_info['on_hit_effect'])
        if 'on_being_hit_effect' in item_info:
            combatant['on_being_hit_effects'].append(item_info['on_being_hit_effect'])

        # Lógica de Revive 
        special_tag = item_info.get('special_effect') 
        on_hit_tag = item_info.get('on_being_hit_effect', {}).get('effect_tag') 
        if (special_tag == 'Revive' or on_hit_tag == 'Revive') and 'Revive' in effect_ruleset:
             combatant['special_effects']['Revive'] = {
                 'used': False,
                 'potency': effect_ruleset['Revive'].get('potency', DEFAULT_REVIVE_POTENCY) 
             }            

    # Carrega os efeitos de sala salvos, se houver
    if saved_effects:        
        combatant['effects'] = copy.deepcopy(saved_effects)

    # Aplica Efeitos de Sala Iniciais 
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        if isinstance(room_rule, dict): 
            if 'effect_to_apply' in room_rule:
                effect_tag = room_rule['effect_to_apply']
                if effect_tag in effect_ruleset:
                    combatant['effects'][effect_tag] = {'duration': -1} # Permanente                

            if 'stat_modifier' in room_rule:
                for stat_key, value in room_rule['stat_modifier'].items():
                    if stat_key == 'damage_modifier':
                        combatant['base_stats']['damage_modifier'] += value
                    elif stat_key in combatant['base_stats']:
                        combatant['base_stats'][stat_key] += value                    

    return combatant

def get_current_stats(
    combatant: Dict[str, Any], 
    catalogs: Dict[str, Any], 
    room_effect_name: Optional[str] = None
) -> Dict[str, float]:
    
    """Calcula stats atuais somando base + efeitos ativos + EFEITOS DE SALA."""
    
    current_stats = combatant['base_stats'].copy()
    effect_ruleset = catalogs.get('effects', {})
    
    # Aplica efeitos próprios (Buffs/Debuffs no personagem)
    for effect_name, effect_data in combatant.get('effects', {}).items():
        rule = effect_ruleset.get(effect_name, {})
        if isinstance(rule, dict) and 'stat_modifier' in rule:
            for stat_key, value in rule['stat_modifier'].items():
                if stat_key in current_stats:
                    current_stats[stat_key] += value

    # Aplica efeitos de sala
    if room_effect_name:
        room_data = catalogs.get('room_effects', {}).get(room_effect_name)
        
        # Verifica se a sala tem modificadores passivos
        if room_data and 'stat_modifier' in room_data:
            for stat_key, value in room_data['stat_modifier'].items():
                current_stats[stat_key] = current_stats.get(stat_key, 0.0) + value

    # Hard Caps de segurança (Usa as Constantes)
    current_stats['accuracy'] = max(CAP_MIN_ACCURACY, current_stats.get('accuracy', 1.0)) 
    current_stats['evasion_chance'] = max(0.0, min(CAP_MAX_EVASION, current_stats.get('evasion_chance', 0.0))) 
    current_stats['crit_chance'] = max(0.0, min(CAP_MAX_CRIT, current_stats.get('crit_chance', 0.0)))
    current_stats['damage_reduction'] = max(0.0, min(CAP_MAX_DMG_REDUCTION, current_stats.get('damage_reduction', 0.0))) 

    return current_stats

def resolve_turn_effects_and_cooldowns(combatant: Dict[str, Any], catalogs: Dict[str, Any], room_effect_name: Optional[str] = None) -> tuple[bool, bool]:
    """
    Aplica DoTs, HoTs, Cooldowns, Efeitos de Sala e verifica morte.    
    """
    
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})
    stats = get_current_stats(combatant, catalogs, room_effect_name)

    is_stunned = False
    died_from_effects = False

    # Efeitos persistentes de sala
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        
        # A) Cura Persistente
        if isinstance(room_rule, dict) and 'persistent_effect' in room_rule:
            persistent_rule = room_rule['persistent_effect']
            if persistent_rule.get('type') == 'Heal_Self':
                heal_pct = persistent_rule.get('heal_per_round', 0.0)
                if heal_pct > 0:
                    heal_amount = combatant['max_hp'] * heal_pct
                    combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

        # B) Aplicação Forçada de Status
        if isinstance(room_rule, dict) and 'effect_to_apply' in room_rule:
            tag_to_apply = room_rule['effect_to_apply']
            effect_config = effect_ruleset.get(tag_to_apply)
            
            if effect_config:
                target_tags = combatant.get('tags', [])
                immune = False
                
                if tag_to_apply == 'Burn' and ('Thermal' in target_tags or 'Fire' in target_tags): 
                    immune = True
                elif tag_to_apply == 'Poison' and ('Undead' in target_tags or 'Construct' in target_tags):
                    immune = True
                
                if not immune:
                    default_duration = effect_config.get('duration', DEFAULT_EFFECT_DURATION)
                    combatant['effects'][tag_to_apply] = {'duration': default_duration}

    # Processamento de efeitos ativos
    active_effects = list(combatant.get('effects', {}).keys())
    
    for effect_name in active_effects:
        if effect_name not in combatant['effects']: continue 

        rule = effect_ruleset.get(effect_name, {})
        if not isinstance(rule, dict): continue

        # DoT
        if rule.get('type') == 'DoT':
            flat_dmg = rule.get('damage_per_turn', 0)
            percent_dmg = combatant['max_hp'] * rule.get('damage_percent', 0.0)
            base_dot = max(flat_dmg, percent_dmg)
            
            final_dot = base_dot * stats.get('dot_potency_modifier', 1.0)
            if final_dot > 0:
                combatant['hp'] -= final_dot

        # HoT
        elif rule.get('type') == 'Heal_Self': 
            potency = rule.get('potency', 0.0)
            if potency > 0:
                heal_amount = combatant['max_hp'] * potency
                combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

        # Control
        if rule.get('type') == 'Control':
            is_stunned = True

        # Decrementa duração
        duration = combatant['effects'][effect_name].get('duration', 0)
        if duration > 0: 
            combatant['effects'][effect_name]['duration'] -= 1
            if combatant['effects'][effect_name]['duration'] <= 0:
                del combatant['effects'][effect_name]
    
    
    hp_before_check = combatant['hp']

    is_permanently_dead = check_for_death_and_revive(combatant, catalogs)
    
    if is_permanently_dead:
        died_from_effects = True
        is_stunned = True 
    elif hp_before_check <= 0:
        died_from_effects = False
        is_stunned = False 

    if not died_from_effects: 
        for skill_name in combatant.get('cooldowns', {}):
            if combatant['cooldowns'][skill_name] > 0:
                combatant['cooldowns'][skill_name] -= 1

    return is_stunned, died_from_effects

def execute_action(
    attacker: Dict[str, Any],
    targets: List[Dict[str, Any]], 
    action_name: str,    
    catalogs: Dict[str, Any],
    room_effect_name: Optional[str] = None,
) -> Tuple[float, bool, str, List[str]]:
    
    # Helper seguro para nomes e HP formatado (inteiro)
    att_name = attacker.get('name', 'Agente')
    att_hp = int(attacker.get('hp', 0))
    att_max = int(attacker.get('max_hp', 1))
    att_status = f"[{att_hp}/{att_max} HP]" # Ex: [150/200 HP]

    log_details = []

    # 1. Rota para Consumíveis
    if action_name == "Use Consumable":                
        r, s, m = use_consumable(attacker, catalogs)
        # Formata melhor a mensagem do consumível
        return r, s, f"{att_name} {m}", []

    # 2. Validações (Logs mais limpos)
    if is_stunned(attacker, catalogs):
        return 0.0, True, f"😵 {att_name} está atordoado e perdeu o turno.", []

    if action_name == 'Wait':
        # "Wait" agora é informativo mas discreto
        return 0.0, True, f"⏸️ {att_name} aguardou. {att_status}", []

    if action_name not in attacker.get('skills', {}): 
        return 0.0, False, f"❌ {att_name}: Skill '{action_name}' inválida.", []
    
    # CD: Mensagem unificada
    current_cd = attacker.get('cooldowns', {}).get(action_name, 0)
    if current_cd > 0:
        return 0.0, False, f"⏳ {att_name}: '{action_name}' em Recarga ({current_cd}t restantes).", []
        
    if not targets:
        return 0.0, False, f"❓ {att_name}: Sem alvos válidos.", []
    
    # --- PREPARAÇÃO DA AÇÃO ---
    skill_info = attacker['skills'][action_name]
    attacker_stats = get_current_stats(attacker, catalogs)
    
    # Identificação de Tipo
    tags = skill_info.get('tags', [])
    is_magic = any(t in ['Magical', 'DoT'] for t in tags)

    # Coleta tags de efeito de sala
    effect_ruleset = catalogs.get('effects', {})
    room_data = catalogs.get('room_effects', {}).get(room_effect_name, {})
    room_tags = room_data.get('tags', [])
    
    # Seleção de Alvo
    skill_target_type = skill_info.get('target', 'Target')
    possible_targets = []
    
    if 'Heal_Self' in tags:
        skill_target_type = 'Self'

    if skill_target_type in ['Target', 'AOE']: 
         possible_targets = [t for t in targets if isinstance(t, dict) and t.get('hp', 0) > 0 and t.get('team') != attacker.get('team')]
    elif skill_target_type == 'Self': 
         possible_targets = [attacker]
         
    is_aoe = skill_target_type == 'AOE'

    if not possible_targets:
        attacker['cooldowns'][action_name] = skill_info.get('cd', 0)
        return 0.0, False, "Alvos inválidos/mortos.", []

    final_targets = possible_targets if is_aoe else [random.choice(possible_targets)]

    # Ícone da Ação
    action_icon = "✨" if is_magic else "⚔️"

    # Resolução da Ação (Dano/Cura Direta)
    for target in final_targets:

        if attacker['hp'] <= 0: 
            break
        if not isinstance(target, dict) or target.get('hp', 0) <= 0: continue 

        target_stats = get_current_stats(target, catalogs)
        base_val = float(skill_info.get('damage', 0)) 
        final_damage_taken = 0.0 
        tar_name = target.get('name', 'Alvo')        
        
        # Se é DANO:
        if base_val > 0: 
            # Hit Check
            hit_chance = attacker_stats.get('accuracy', 1.0) - target_stats.get('evasion_chance', 0.0)            
            hit_chance = max(HIT_CHANCE_FLOOR, min(hit_chance, HIT_CHANCE_CEILING))

            roll = random.random()

            if roll < hit_chance: 
                global_mod = attacker_stats.get('damage_modifier', 1.0)
                type_mod = attacker_stats.get('magic_damage_modifier', 1.0) if is_magic else attacker_stats.get('physical_damage_modifier', 1.0)

                # Modificadores de Sala                
                room_dmg_mod = 1.0
                if is_magic and 'Magical' in room_tags:                    
                    room_dmg_mod += room_data.get('stat_modifier', {}).get('damage_modifier', 0.0)
                elif not is_magic and 'Physical' in room_tags:
                    room_dmg_mod += room_data.get('stat_modifier', {}).get('damage_modifier', 0.0)
                
                # Defesa
                defense_mod = (1.0 - target_stats.get('damage_reduction', 0.0)) * target_stats.get('defense_taken', 1.0)
                defense_mod = max(MIN_DEFENSE_MULTIPLIER, defense_mod) 
                
                # Crit
                is_crit = random.random() < attacker_stats.get('crit_chance', 0.05)
                crit_mod = CRIT_MULTIPLIER_DEFAULT if is_crit else 1.0
                
                # Fórmula Final
                stat_damage = attacker_stats.get('damage', 0.0) 
                flat_bonus = attacker_stats.get('flat_damage_bonus', 0.0)
                total_flat = stat_damage + flat_bonus
                total_damage = (base_val + total_flat) * global_mod * type_mod * crit_mod * room_dmg_mod
                final_damage_taken = max(0.0, total_damage * defense_mod)

                target['hp'] -= final_damage_taken

                # Life Drain
                if 'Life_Drain' in tags and 'Life_Drain' in effect_ruleset:
                    potency = effect_ruleset['Life_Drain'].get('potency', 0.0)
                    heal = final_damage_taken * potency
                    attacker['hp'] = min(attacker['max_hp'], attacker['hp'] + heal)

                # On-Hit Effects (Do Atacante)
                for on_hit in attacker.get('on_hit_effects', []):
                    if random.random() < on_hit.get('chance', 0.0):
                        tag = on_hit.get('effect_tag')
                        if tag and tag in effect_ruleset:
                            rule = effect_ruleset[tag]
                            if 'duration' in rule: 
                                target['effects'][tag] = {'duration': rule['duration']}

                # On-Being-Hit Effects (Do Alvo)
                if target['hp'] > 0: 
                    for on_being_hit in target.get('on_being_hit_effects', []):
                        if random.random() < on_being_hit.get('chance', 0.0):
                            tag = on_being_hit.get('effect_tag')
                            if tag in effect_ruleset:
                                rule = effect_ruleset[tag]
                                if tag == 'Reflect': 
                                    reflected_damage = final_damage_taken * rule.get('potency', 0.0)
                                    attacker['hp'] -= reflected_damage
                                elif 'duration' in rule: 
                                    attacker['effects'][tag] = {'duration': rule['duration']}
                
                # Log de Dano
                tar_hp_new = int(target['hp'])
                tar_hp_old = int(target['hp'] + final_damage_taken)
                
                crit_str = " 💥 CRÍTICO!" if is_crit else ""
                dmg_str = f"{int(final_damage_taken)}"
                                
                log_details.append(
                    f"   🎯 [HIT] {tar_name}: -{dmg_str} HP{crit_str}. (HP: {tar_hp_old} -> {tar_hp_new})"
                )

            else:
                log_details.append(f"   💨 [MISS] {tar_name} esquivou! ({int(roll*100)}% vs {int(hit_chance*100)}%)")

        # Se é CURA:
        elif base_val < 0: 
            flat_heal = abs(base_val)
            percent_heal = 0.0
            
            if 'Heal_Self' in tags and 'Heal_Self' in effect_ruleset:
                potency = effect_ruleset['Heal_Self'].get('potency', 0.0)
                percent_heal = target['max_hp'] * potency   

            room_heal_mod = 1.0
            if 'Healing_Debuff' in room_tags:
                room_heal_mod = ROOM_HEAL_DEBUFF_MOD 

            final_heal = (flat_heal + percent_heal) * room_heal_mod 
                        
            target['hp'] = min(target['max_hp'], target['hp'] + final_heal)

            heal_val = int(final_heal)
            tar_hp_new = int(target['hp'])
            
            log_details.append(
                f"   💚 [HEAL] {tar_name}: +{heal_val} HP. (HP Atual: {tar_hp_new})"
            )

    # Aplica Efeitos Secundários (Burn, Poison, Sunder, etc) CORRIGIDO
    for tag in skill_info.get('tags', []):
        if tag in effect_ruleset:
            rule = effect_ruleset[tag]
            
            if isinstance(rule, dict) and 'duration' in rule:                
                effect_target_type = rule.get('target', 'Target')
                apply_list = final_targets if effect_target_type != 'Self' else [attacker]
                
                for et in apply_list:
                    # Chega a imunidade para cada alvo
                    et_tags = et.get('tags', [])
                    if tag == 'Burn' and ('Thermal' in et_tags or 'Fire' in et_tags): continue
                    if tag == 'Poison' and ('Undead' in et_tags or 'Construct' in et_tags): continue
                    if tag == 'Sunder' and ('Skeleton' in et_tags or 'Construct' in et_tags): continue

                    if isinstance(et, dict) and et.get('hp', 0) > 0: 
                        et['effects'][tag] = {'duration': rule['duration']}
                        log_details.append(f"  > [EFFECT] {tag} aplicado em {et.get('name', 'Alvo')}")

    # Set cooldown
    cd_val = skill_info.get('cd', 0)
    attacker['cooldowns'][action_name] = cd_val
    
    # Pega as skills equipadas (deck) para mostrar o estado atual de cada uma
    deck = attacker.get('skills', [])
    current_cds = attacker.get('cooldowns', {})
    
    # Monta uma lista. Ex: "Quick Strike:0 | Heavy Blow:2"
    # Filtramos 'Wait' para não poluir
    cd_list_str = " | ".join([f"{s}:{current_cds.get(s, 0)}" for s in deck if s != 'Wait'])
    
    # Adiciona na mensagem principal
    main_msg = f"{action_icon} {att_name} usou {action_name}! {att_status} [CDs: {cd_list_str}]"

    return 0.0, True, main_msg, log_details

def is_stunned(combatant: dict, catalogs: dict) -> bool:
    """Retorna True se o combatente tiver qualquer efeito do tipo 'Control'."""
    effect_ruleset = catalogs.get('effects', {})
    
    for effect_name in combatant.get('effects', {}):
        rule = effect_ruleset.get(effect_name)
        if rule and rule.get('type') == 'Control':
            return True
    return False

def check_for_death_and_revive(combatant: Dict[str, Any], catalogs: Dict[str, Any]) -> bool:
    """Lógica Original de Morte e Revive"""
    if combatant.get('hp', 0) <= 0:
        revive_status = combatant.get('special_effects', {}).get('Revive')
        if revive_status and not revive_status.get('used', True): 
            revive_potency = revive_status.get('potency', DEFAULT_REVIVE_POTENCY)
            combatant['hp'] = combatant['max_hp'] * revive_potency
            combatant['special_effects']['Revive']['used'] = True            
            return False 
        else:
            return True 
    return False

def use_consumable(agent: dict, catalogs: dict) -> tuple[float, bool, str]:
    """
    Consome o item no slot 'Consumable' do agente.
    
    Retorna uma tupla:
    (valor_para_reward, sucesso, mensagem_de_log)
    """
    item_name = agent['equipment'].get('Consumable')
    
    # Checa se tem item
    if not item_name:
        return REW_ITEM_FAIL, False, "Tentou usar consumível mas o slot estava vazio."
        
    # Pega dados
    details = catalogs['equipment'].get(item_name, {})
    effect_type = details.get('effect')
    value = details.get('value', 0)
    duration = details.get('duration', 0)
    
    reward_base = 0.0
    success = False
    log_msg = ""

    # Processa Efeitos
    if effect_type == 'heal':
        old_hp = agent['hp']
        
        # Só cura se precisar
        if old_hp < agent['max_hp']:
            heal_amount = value
            # Cura (respeitando Max HP)
            agent['hp'] = min(agent['max_hp'], old_hp + heal_amount)
            recovered = agent['hp'] - old_hp
            
            # Reward por cura efetiva
            reward_base = REW_ITEM_HEAL_BASE + (recovered * REW_ITEM_HEAL_SCALAR) 
            log_msg = f"Bebeu '{item_name}'. Recuperou {recovered} HP."
            success = True
        else:
            reward_base = REW_ITEM_WASTE # Pequena penalidade por desperdício
            log_msg = f"Tentou beber '{item_name}' mas já estava com vida cheia."
            success = False # Falhou em usar (não gasta o item)
            return reward_base, False, log_msg

    elif effect_type == 'heal_full':
        # Elixir of Rejuvenation
        old_hp = agent['hp']
        if old_hp < agent['max_hp']:
            agent['hp'] = agent['max_hp']
            recovered = agent['hp'] - old_hp
            
            reward_base = REW_ITEM_FULL_HEAL_BASE + (recovered * REW_ITEM_HEAL_SCALAR)
            log_msg = f"Bebeu '{item_name}'. Vida totalmente restaurada (+{recovered})."
            success = True
        else:
            return REW_ITEM_WASTE, False, "Vida já está cheia."

    elif effect_type in ['buff_damage', 'buff_defense']:
        # Buffs Genéricos
        buff_key = 'damage_bonus' if effect_type == 'buff_damage' else 'defense_bonus'
        buff_name = f"Buff_{item_name}"
        
        # Garante estrutura de effects
        if 'effects' not in agent: agent['effects'] = {}
            
        agent['effects'][buff_name] = {'duration': duration, buff_key: value}
        
        reward_base = REW_ITEM_BUFF_BASE
        log_msg = f"Bebeu '{item_name}'. {buff_key} +{value} por {duration} turnos."
        success = True
        
    elif effect_type == 'god_mode':
        # Ambrosia (Lendário)
        buff_name = "God_Mode"
        if 'effects' not in agent: agent['effects'] = {}
            
        agent['effects'][buff_name] = {
            'duration': duration, 
            'damage_bonus': 50, 
            'defense_bonus': 50,
            'accuracy_bonus': 1.0
        }
        agent['hp'] = agent['max_hp'] # Cura tudo também
        
        reward_base = REW_ITEM_GOD_MODE
        log_msg = f"Consumiu '{item_name}'. PODER ABSOLUTO por {duration} turnos!"
        success = True

    # Consome o item
    if success:
        agent['equipment']['Consumable'] = None
        
    return reward_base, success, log_msg
