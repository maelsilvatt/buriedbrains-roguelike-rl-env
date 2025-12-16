# buriedbrains/combat.py
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

    # Calcula o Scaling de Nível         
    level_scaling = 1.0 + (level // 10) * 0.15
        
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
            'damage_modifier': 1.0 * level_scaling,                        
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
                 'potency': effect_ruleset['Revive'].get('potency', 0.5) 
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
                    # Ajuste fino: Aplica modifiers genéricos aos específicos se necessário
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
            
            # Checagem de segurança: 
            # Alguns efeitos de sala são condicionais (ex: +Dano Mágico).                        
            for stat_key, value in room_data['stat_modifier'].items():
                # Se o stat existe na ficha do personagem, aplica o bônus da sala
                # Ex: Soma 0.25 em evasion_chance
                current_stats[stat_key] = current_stats.get(stat_key, 0.0) + value

    # Hard Caps de segurança
    current_stats['accuracy'] = max(0.1, current_stats.get('accuracy', 1.0)) 
    current_stats['evasion_chance'] = max(0.0, min(0.75, current_stats.get('evasion_chance', 0.0))) 
    current_stats['crit_chance'] = max(0.0, min(1.0, current_stats.get('crit_chance', 0.0)))
    current_stats['damage_reduction'] = max(0.0, min(0.70, current_stats.get('damage_reduction', 0.0))) 

    return current_stats

def resolve_turn_effects_and_cooldowns(combatant: Dict[str, Any], catalogs: Dict[str, Any], room_effect_name: Optional[str] = None) -> tuple[bool, bool]:
    """
    Aplica DoTs, HoTs, Cooldowns, Efeitos de Sala e verifica morte.    
    """
    
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})
    stats = get_current_stats(combatant, catalogs, room_effect_name) # Atualizado para ler stats da sala

    is_stunned = False
    died_from_effects = False

    # Efeitos persistentes de sala
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        
        # A) Cura Persistente (Ex: Consecrated Ground)
        if isinstance(room_rule, dict) and 'persistent_effect' in room_rule:
            persistent_rule = room_rule['persistent_effect']
            if persistent_rule.get('type') == 'Heal_Self':
                heal_pct = persistent_rule.get('heal_per_round', 0.0)
                if heal_pct > 0:
                    heal_amount = combatant['max_hp'] * heal_pct
                    combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

        # B) Aplicação Forçada de Status (Ex: Heat -> Burn, Fog -> Blind)
        # Garante que o ambiente "ataque" o combatente reaplicando o debuff
        if isinstance(room_rule, dict) and 'effect_to_apply' in room_rule:
            tag_to_apply = room_rule['effect_to_apply'] # Ex: 'Burn', 'Slow'
            effect_config = effect_ruleset.get(tag_to_apply)
            
            if effect_config:
                # Checagem de Imunidades Básicas (Para não queimar quem é de fogo)
                target_tags = combatant.get('tags', [])
                immune = False
                
                # Fogo não queima Fogo lol
                if tag_to_apply == 'Burn' and ('Thermal' in target_tags or 'Fire' in target_tags): 
                    immune = True
                # Veneno não afeta Mortos-Vivos
                elif tag_to_apply == 'Poison' and ('Undead' in target_tags or 'Construct' in target_tags):
                    immune = True
                
                if not immune:
                    # Aplica ou renova a duração do efeito
                    # Se o efeito já existe, isso "reseta" o contador para o máximo da duração
                    default_duration = effect_config.get('duration', 2)
                    combatant['effects'][tag_to_apply] = {'duration': default_duration}
                    # # print(f"DEBUG: Ambiente reaplicou {tag_to_apply}")

    # Processamento de efeitos ativos no combatente
    # Cria uma lista snapshot para iterar, pois podemos deletar chaves
    active_effects = list(combatant.get('effects', {}).keys())
    
    for effect_name in active_effects:
        if effect_name not in combatant['effects']: continue 

        rule = effect_ruleset.get(effect_name, {})
        if not isinstance(rule, dict): continue

        # DoT (Damage over Time)
        if rule.get('type') == 'DoT':
            flat_dmg = rule.get('damage_per_turn', 0)
            percent_dmg = combatant['max_hp'] * rule.get('damage_percent', 0.0)
            base_dot = max(flat_dmg, percent_dmg)
            
            final_dot = base_dot * stats.get('dot_potency_modifier', 1.0)
            if final_dot > 0:
                combatant['hp'] -= final_dot
                # # print(f"DEBUG: {combatant['name']} sofreu {final_dot} de {effect_name}")

        # HoT (Heal over Time)
        elif rule.get('type') == 'Heal_Self': 
            potency = rule.get('potency', 0.0)
            if potency > 0:
                heal_amount = combatant['max_hp'] * potency
                combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

        # Checa se está stunado
        if rule.get('type') == 'Control':
            is_stunned = True

        # Decrementa duração do efeito
        duration = combatant['effects'][effect_name].get('duration', 0)
        if duration > 0: 
            combatant['effects'][effect_name]['duration'] -= 1
            if combatant['effects'][effect_name]['duration'] <= 0:
                del combatant['effects'][effect_name]
    
    
    hp_before_check = combatant['hp']

    # Confirma morte e aplica Revive se disponível
    is_permanently_dead = check_for_death_and_revive(combatant, catalogs)
    
    if is_permanently_dead:
        died_from_effects = True
        is_stunned = True # Morto não age
    elif hp_before_check <= 0:
        # Para o caso onde revive foi acionado
        died_from_effects = False
        is_stunned = False # Revive limpa o Stun

    # Decrementa os cooldowns
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
) -> None:
    """
    Executa Skill.    
    """        

    # Se estiver stunado, perde a vez, mas conta como turno "jogado"
    if is_stunned(attacker, catalogs):
        # print(f"🚫 {attacker['name']} está ATORDOADO e perdeu a vez!")
        return True 

    if action_name == 'Wait' or not targets: return
        
    # Carrega dados da sala
    effect_ruleset = catalogs.get('effects', {})
    room_data = catalogs.get('room_effects', {}).get(room_effect_name, {})
    room_tags = room_data.get('tags', [])
    
    if action_name not in attacker.get('skills', {}): return

    if attacker.get('cooldowns', {}).get(action_name, 0) > 0:
        # print(f"   > Ação falhou: '{action_name}' está em cooldown ({attacker['cooldowns'][action_name]} turnos restantes).")
        return False # FALHOU: Estava em cooldown

    # --- DEBUG 1: SNAPSHOT DA DECISÃO ---
    # Captura o estado dos CDs ANTES de processar, para ver se a IA escolheu uma skill válida
    cd_snapshot = attacker.get('cooldowns', {}).copy()    

    if attacker.get('cooldowns', {}).get(action_name, 0) > 0:     
        # print(f"[DEBUG-COMBAT] ⚠️ {attacker['name']} tentou usar '{action_name}' mas estava em CD! {cd_snapshot}")
        return 
    
    skill_info = attacker['skills'][action_name]
    attacker_stats = get_current_stats(attacker, catalogs)

    # --- DEBUG 2: INÍCIO DO TURNO ---
    # Diferencia visualmente Agente (Team 1) de Inimigo (Team 2)    
    actor_icon = "🤖 [AGENTE]" if attacker.get('team') == 1 else "👾 [INIMIGO]"
    # print(f"\n{actor_icon} {attacker['name']} (Lvl {attacker.get('level')}) usa: {action_name}")
    if attacker.get('team') == 2:
        # print(f"   > Cooldowns no momento da escolha: {cd_snapshot}")
        pass

    # --- Identificação de Tipo ---
    tags = skill_info.get('tags', [])
    is_magic = any(t in ['Magical', 'Fire', 'Ice', 'Arcane', 'DoT'] for t in tags)
    is_aoe = 'AOE' in tags
    
    # --- Seleção de Alvo ---
    skill_target_type = skill_info.get('target', 'Target')
    possible_targets = []
    
    if 'Heal_Self' in tags:
        skill_target_type = 'Self'

    if skill_target_type in ['Target', 'AOE']: 
         possible_targets = [t for t in targets if isinstance(t, dict) and t.get('hp', 0) > 0 and t.get('team') != attacker.get('team')]
    elif skill_target_type == 'Self': 
         possible_targets = [attacker]
         
    if not possible_targets:
        attacker['cooldowns'][action_name] = skill_info.get('cd', 0)
        # print(f"   > Ação falhou: Sem alvos válidos.")
        return

    final_targets = possible_targets if is_aoe else [random.choice(possible_targets)]

    # --- Resolução da Ação ---
    for target in final_targets:

        if attacker['hp'] <= 0: 
            break
        if not isinstance(target, dict) or target.get('hp', 0) <= 0: continue 

        target_stats = get_current_stats(target, catalogs)
        base_val = float(skill_info.get('damage', 0)) 
        final_damage_taken = 0.0 
        
        # --- DANO (Positivo) ---
        if base_val > 0: 
            # Hit Check
            hit_chance = attacker_stats.get('accuracy', 1.0) - target_stats.get('evasion_chance', 0.0)            
                
            # Tem chance mínima de 5% de acertar e máxima de 95%
            hit_chance = max(0.05, min(hit_chance, 0.95))

            if random.random() < hit_chance: 
                # Modificadores
                global_mod = attacker_stats.get('damage_modifier', 1.0)
                type_mod = attacker_stats.get('magic_damage_modifier', 1.0) if is_magic else attacker_stats.get('physical_damage_modifier', 1.0)

                # Modificadores de Sala                
                room_dmg_mod = 1.0
                
                # Bônus para Magia
                if is_magic and 'Magical' in room_tags:                    
                    mod_val = room_data.get('stat_modifier', {}).get('damage_modifier', 0.0)
                    room_dmg_mod += mod_val
                    # print(f"      ⚡ Sala Arcana! Dano Mágico amplificado em {mod_val*100:.0f}%")

                # Bônus para Físico
                elif not is_magic and 'Physical' in room_tags:
                    mod_val = room_data.get('stat_modifier', {}).get('damage_modifier', 0.0)
                    room_dmg_mod += mod_val
                    # print(f"      ⚔️ Campo de Força! Dano Físico amplificado em {mod_val*100:.0f}%")
                
                # Defesa
                defense_mod = (1.0 - target_stats.get('damage_reduction', 0.0)) * target_stats.get('defense_taken', 1.0)
                defense_mod = max(0.1, defense_mod) 
                
                # Crit
                is_crit = random.random() < attacker_stats.get('crit_chance', 0.05)
                crit_mod = 2.0 if is_crit else 1.0
                crit_text = " (CRÍTICO!)" if is_crit else ""
                
                # Fórmula Final
                stat_damage = attacker_stats.get('damage', 0.0) # Dano nativo do monstro
                flat_bonus = attacker_stats.get('flat_damage_bonus', 0.0)
                total_flat = stat_damage + flat_bonus
                total_damage = (base_val + total_flat) * global_mod * type_mod * crit_mod * room_dmg_mod
                final_damage_taken = max(0.0, total_damage * defense_mod)

                target['hp'] -= final_damage_taken

                # --- DEBUG 3: MATEMÁTICA DO DANO ---
                # print(f"   🎯 ACERTOU{crit_text} em {target['name']}!")
                # print(f"      Matemática: (Base {base_val} + Flat {total_flat:.1f}) x Mod {global_mod:.2f} = {((base_val+flat_bonus)*global_mod):.1f} Bruto")
                # print(f"      Defesa Alvo: {defense_mod:.2f} -> Dano Final: {final_damage_taken:.1f}")
                # print(f"      HP Restante: {target['hp']:.1f}/{target['max_hp']:.1f}")

                # Life Drain
                if 'Life_Drain' in tags and 'Life_Drain' in effect_ruleset:
                    potency = effect_ruleset['Life_Drain'].get('potency', 0.0)
                    heal = final_damage_taken * potency
                    attacker['hp'] = min(attacker['max_hp'], attacker['hp'] + heal)
                    # print(f"      🩸 Life Drain: Curou {heal:.1f} HP do atacante.")

                # On-Hit Effects
                for on_hit in attacker.get('on_hit_effects', []):
                    if random.random() < on_hit.get('chance', 0.0):
                        tag = on_hit.get('effect_tag')
                        if tag and tag in effect_ruleset:
                            rule = effect_ruleset[tag]
                            if 'duration' in rule: 
                                target['effects'][tag] = {'duration': rule['duration']}
                                # print(f"      ✨ Aplicou Efeito On-Hit: {tag}")

                # On-Being-Hit Effects
                if target['hp'] > 0: 
                    for on_being_hit in target.get('on_being_hit_effects', []):
                        if random.random() < on_being_hit.get('chance', 0.0):
                            tag = on_being_hit.get('effect_tag')
                            if tag in effect_ruleset:
                                rule = effect_ruleset[tag]
                                if tag == 'Reflect': 
                                    reflected_damage = final_damage_taken * rule.get('potency', 0.0)
                                    attacker['hp'] -= reflected_damage
                                    # print(f"      🛡️ REFLECT! Atacante tomou {reflected_damage:.1f} de volta.")
                                elif 'duration' in rule: 
                                    attacker['effects'][tag] = {'duration': rule['duration']}

            else:
                # print(f"   💨 ERROU o ataque em {target['name']}! (Chance: {hit_chance*100:.1f}%)")
                pass

        # Cura
        elif base_val < 0: 
            flat_heal = abs(base_val)
            percent_heal = 0.0
            
            if 'Heal_Self' in tags and 'Heal_Self' in effect_ruleset:
                potency = effect_ruleset['Heal_Self'].get('potency', 0.0)
                percent_heal = target['max_hp'] * potency   

            # Efeito de sala que reduz cura (Weakening Field)
            room_heal_mod = 1.0
            
            if 'Healing_Debuff' in room_tags:
                room_heal_mod = 0.5 # Reduz eficácia em 50%
                # print(f"      💀 Aura Profana reduz a cura pela metade!")

            final_heal = (flat_heal + percent_heal) * room_heal_mod 
                        
            target['hp'] = min(target['max_hp'], target['hp'] + final_heal)
            # print(f"   💚 CUROU {final_heal:.1f} HP. HP Atual: {target['hp']:.1f}")       

    # Aplica Efeitos Secundários (Burn, Poison, Sunder, etc)
    for tag in skill_info.get('tags', []):
        if tag in effect_ruleset:
            rule = effect_ruleset[tag]

            target_tags = target.get('tags', [])
            
            # Fogo não queima Fogo
            if tag == 'Burn' and ('Thermal' in target_tags or 'Fire' in target_tags):
                # print(f"      🚫 {target['name']} é IMUNE a Burn!")
                continue # Pula a aplicação deste efeito
            
            # Veneno não afeta Mortos-Vivos
            if tag == 'Poison' and ('Undead' in target_tags or 'Construct' in target_tags):
                 # print(f"      🚫 {target['name']} é IMUNE a Poison!")
                 continue
                 
            # Sangramento (Bleed/Sunder) não afeta quem não tem sangue
            if tag == 'Sunder' and ('Skeleton' in target_tags or 'Construct' in target_tags):
                 # print(f"      🚫 {target['name']} é IMUNE a Sunder!")
                 continue
                        
            if isinstance(rule, dict) and 'duration' in rule:                 
                effect_target_type = rule.get('target', 'Target')
                
                # Define quem recebe o efeito (Alvo ou Quem castou)
                apply_list = final_targets if effect_target_type != 'Self' else [attacker]
                
                for et in apply_list:
                    if isinstance(et, dict) and et.get('hp', 0) > 0: 
                        # Aplica ou renova o efeito
                        # Se já tiver o efeito, renova a duração para o máximo
                        et['effects'][tag] = {'duration': rule['duration']}
                        # print(f"      ✨ Aplicou Tag: {tag}")

    # Set Cooldown
    cd_val = skill_info.get('cd', 0)
    attacker['cooldowns'][action_name] = cd_val
    # print(f"   🕒 Cooldown aplicado: {action_name} -> {cd_val}")
    # print("-" * 60)

def is_stunned(combatant: dict, catalogs: dict) -> bool:
    """Retorna True se o combatente tiver qualquer efeito do tipo 'Control'."""
    effect_ruleset = catalogs.get('effects', {})
    
    for effect_name in combatant.get('effects', {}):
        rule = effect_ruleset.get(effect_name)
        # Se achou um efeito e o tipo dele é Control (Stun, Freeze, etc)
        if rule and rule.get('type') == 'Control':
            return True
    return False

def check_for_death_and_revive(combatant: Dict[str, Any], catalogs: Dict[str, Any]) -> bool:
    """Lógica Original de Morte e Revive"""
    if combatant.get('hp', 0) <= 0:
        revive_status = combatant.get('special_effects', {}).get('Revive')
        if revive_status and not revive_status.get('used', True): 
            revive_potency = revive_status.get('potency', 0.5)
            combatant['hp'] = combatant['max_hp'] * revive_potency
            combatant['special_effects']['Revive']['used'] = True            
            return False 
        else:
            return True 
    return False