# buriedbrains/combat.py
import numpy as np
import copy
import random
from typing import Dict, Any, List, Optional

def initialize_combatant(
    name: str, 
    hp: int, 
    equipment: list, 
    skills: list, 
    team: int, 
    catalogs: Dict[str, Any],
    room_effect_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepara a 'ficha de personagem' de um combatente para a batalha, aplicando
    efeitos passivos de equipamentos e efeitos de sala.
   
    """
    skill_catalog = catalogs.get('skills', {})
    equipment_catalog = catalogs.get('equipment', {})
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})

    combatant = {
        'name': name,
        'hp': hp,
        'max_hp': hp,
        'team': team,
        'skills': {s_name: skill_catalog.get(s_name, {}) for s_name in skills},
        'equipment': {e_name: equipment_catalog.get(e_name, {}) for e_name in equipment},
        'cooldowns': {s: 0 for s in skills},
        'effects': {},
        'base_stats': {
            'damage_modifier': 1.0, 'flat_damage_bonus': 0, 'damage_reduction': 0.0,
            'crit_chance': 0.0, 'accuracy': 1.0, 'evasion_chance': 0.0,
            'dot_potency_modifier': 1.0
        },
        'on_hit_effects': [],
        'on_being_hit_effects': [],
        'special_effects': {}
    }

    # Aplica bônus de equipamento
    for item_info in combatant['equipment'].values():
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
        if item_info.get('special_effect') == 'Revive':
            combatant['special_effects']['Revive'] = {'used': False, 'potency': effect_ruleset.get('Revive', {}).get('potency', 0)}

    # Aplica efeitos permanentes da sala
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        if 'effect_to_apply' in room_rule and room_rule['effect_to_apply'] in effect_ruleset:
            combatant['effects'][room_rule['effect_to_apply']] = {'duration': -1} # Duração "permanente"
        if 'stat_modifier' in room_rule:
            for stat, value in room_rule['stat_modifier'].items():
                if stat in combatant['base_stats']:
                    combatant['base_stats'][stat] += value
                    
    return combatant

def get_current_stats(combatant: Dict[str, Any], effect_ruleset: Dict[str, Any]) -> Dict[str, float]:
    """Calcula os atributos atuais de um combatente, aplicando modificadores de efeitos ativos."""
    current_stats = combatant['base_stats'].copy()
    for effect_name in combatant.get('effects', {}):
        rule = effect_ruleset.get(effect_name, {})
        if 'stat_modifier' in rule:
            for stat_key, value in rule['stat_modifier'].items():
                if stat_key in current_stats:
                    current_stats[stat_key] += value
    return current_stats

def resolve_turn_effects_and_cooldowns(combatant: Dict[str, Any], catalogs: Dict[str, Any]):
    """
    Aplica todos os efeitos de início de turno (DoTs, HoTs) e reduz os cooldowns.
    Retorna True se o combatente estiver atordoado, False caso contrário.
    """
    effect_ruleset = catalogs.get('effects', {})
    stats = get_current_stats(combatant, effect_ruleset)
    is_stunned = False

    # Lógica de efeitos persistentes da sala (do notebook)
    room_effect_catalog = catalogs.get('room_effects', {})
    # A classe do ambiente principal precisará passar o 'room_effect_name'
    room_effect_name = combatant.get('current_room_effect') # Exemplo de como poderia ser passado
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        if 'persistent_effect' in room_rule:
            persistent_rule = room_rule['persistent_effect']
            if persistent_rule.get('type') == 'Heal_Self':
                heal_amount = combatant['max_hp'] * persistent_rule.get('heal_per_round', 0)
                combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

    # Aplica efeitos
    active_effects = list(combatant.get('effects', {}).keys())
    for effect_name in active_effects:
        if effect_name not in combatant['effects']: continue
        
        rule = effect_ruleset.get(effect_name, {})
        if rule.get('type') == 'DoT':
            dot_damage = rule.get('damage_per_turn', 0) * stats.get('dot_potency_modifier', 1.0)
            combatant['hp'] -= dot_damage
        elif rule.get('type') == 'Heal_Self':
            heal = rule.get('potency', 0) * combatant['max_hp']
            combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal)
        
        if rule.get('type') == 'Control' and effect_name in ['Stun', 'Fear']:
            is_stunned = True

        # Decrementa duração
        duration = combatant['effects'][effect_name]['duration']
        if duration != -1:
            combatant['effects'][effect_name]['duration'] -= 1
            if combatant['effects'][effect_name]['duration'] <= 0:
                del combatant['effects'][effect_name]
    
    # Reduz cooldowns
    for skill in combatant['cooldowns']:
        if combatant['cooldowns'][skill] > 0:
            combatant['cooldowns'][skill] -= 1

    return is_stunned

def execute_action(
    attacker: Dict[str, Any], 
    targets: List[Dict[str, Any]], 
    action_name: str, 
    catalogs: Dict[str, Any]
):
    """Executa uma única ação (habilidade) de um atacante em um ou mais alvos."""
    if action_name == 'Wait' or not targets:
        return

    effect_ruleset = catalogs.get('effects', {})
    skill_info = attacker['skills'][action_name]
    attacker_stats = get_current_stats(attacker, effect_ruleset)

    # Lógica de seleção de alvos AOE vs Alvo Único (do notebook) - Está correta!
    is_aoe = 'AOE' in skill_info.get('tags', [])
    
    possible_targets = [t for t in targets if t['hp'] > 0]
    
    if not possible_targets:
        return

    if is_aoe:
        final_targets = possible_targets
    else:
        final_targets = [random.choice(possible_targets)]

    for defender in final_targets:
        if defender['hp'] <= 0: continue

        defender_stats = get_current_stats(defender, effect_ruleset)
        base_damage = skill_info.get('damage', 0)

        if base_damage > 0: # Lógica de dano
            hit_chance = max(0.05, min(attacker_stats.get('accuracy', 1.0) - defender_stats.get('evasion_chance', 0.0), 0.95))
            if np.random.rand() < hit_chance:
                damage_mod = attacker_stats.get('damage_modifier', 1.0)
                defense_mod = 1.0 - defender_stats.get('damage_reduction', 0.0)
                is_crit = np.random.rand() < attacker_stats.get('crit_chance', 0.0)
                crit_mod = 2.0 if is_crit else 1.0
                
                final_damage = (base_damage + attacker_stats.get('flat_damage_bonus', 0)) * damage_mod * crit_mod
                final_damage_taken = final_damage * defense_mod
                defender['hp'] -= final_damage_taken
                
                # Aplica efeitos ON-HIT do atacante
                for on_hit in attacker['on_hit_effects']:
                    if np.random.rand() < on_hit['chance']:
                        tag = on_hit['effect_tag']
                        if tag in effect_ruleset:
                            defender['effects'][tag] = {'duration': effect_ruleset[tag]['duration']}
                
                # Aplica efeitos ON-BEING-HIT do defensor
                if defender['hp'] > 0:
                    for on_being_hit in defender['on_being_hit_effects']:
                        if np.random.rand() < on_being_hit['chance']:
                            tag = on_being_hit['effect_tag']
                            if tag == 'Reflect':
                                attacker['hp'] -= final_damage_taken * effect_ruleset['Reflect']['potency']
                            elif tag in effect_ruleset:
                                attacker['effects'][tag] = {'duration': effect_ruleset[tag]['duration']}

        # >> ADICIONADO: Lógica para curas (dano negativo) <<
        elif base_damage < 0:
            # Curas geralmente têm como alvo o próprio usuário ou aliados.
            # Esta lógica simples assume que a cura é sempre no atacante.
            heal_amount = abs(base_damage)
            attacker['hp'] = min(attacker['max_hp'], attacker['hp'] + heal_amount)

    # >> MOVIDO PARA FORA DO LOOP 'for defender': Aplica efeitos da skill e cooldown UMA VEZ <<
    # Isso corrige o bug de cooldown múltiplo em AOE e permite buffs de dano zero.

    # Aplica efeitos da habilidade (buffs, debuffs, etc.)
    for tag in skill_info.get('tags', []):
        if tag in effect_ruleset:
            rule = effect_ruleset[tag]
            
            # Define os alvos para o efeito (pode ser o próprio atacante, um inimigo ou todos)
            if rule.get('target') == 'Self':
                effect_targets = [attacker]
            else:
                effect_targets = final_targets # Aplica a todos os alvos da habilidade (1 ou vários)

            for target in effect_targets:
                target['effects'][tag] = {'duration': rule['duration']}
    
    # Entra em cooldown
    attacker['cooldowns'][action_name] = skill_info.get('cd', 0)

def check_for_death_and_revive(combatant: Dict[str, Any]):
    """Verifica se um combatente morreu e tenta usar a habilidade Revive se disponível."""
    if combatant['hp'] <= 0:
        revive_special = combatant['special_effects'].get('Revive')
        if revive_special and not revive_special.get('used'):
            revive_potency = revive_special.get('potency', 0)
            combatant['hp'] = combatant['max_hp'] * revive_potency
            revive_special['used'] = True
            return False # Não morreu permanentemente
        return True # Morreu permanentemente
    return False