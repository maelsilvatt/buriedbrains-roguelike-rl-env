# buriedbrains/combat.py
import random
from typing import Dict, Any, List, Optional      


def initialize_combatant(
    name: str,
    hp: int,
    equipment: list, # List of equipment names
    skills: list,    # List of skill names
    team: int,
    catalogs: Dict[str, Any],
    room_effect_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepara a 'ficha de personagem' based on notebook logic.
    Applies passive equipment bonuses and initial room effects.
    """
    skill_catalog = catalogs.get('skills', {})
    equipment_catalog = catalogs.get('equipment', {})
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})

    combatant = {
        'name': name,
        'hp': float(hp), # Use float for potentially fractional heals/dots later
        'max_hp': float(hp),
        'team': team,
        # Store full skill/equipment info directly
        'skills': {s_name: skill_catalog.get(s_name, {}).copy() for s_name in skills if s_name in skill_catalog},
        'equipment': {e_name: equipment_catalog.get(e_name, {}).copy() for e_name in equipment if e_name in equipment_catalog},
        'cooldowns': {s: 0 for s in skills},
        'effects': {}, # Active status effects {effect_tag: {'duration': X}}
        # Base stats before temporary effects
        'base_stats': {
            'damage_modifier': 1.0, 'flat_damage_bonus': 0.0, 'damage_reduction': 0.0,
            'crit_chance': 0.0, 'accuracy': 1.0, 'evasion_chance': 0.0,
            'dot_potency_modifier': 1.0,            
            'speed': 1.0, # Example base speed
            'defense_taken': 1.0, # Multiplier for damage taken after reduction
        },
        'on_hit_effects': [], # List of {'chance': X, 'effect_tag': Y}
        'on_being_hit_effects': [], # List of {'chance': X, 'effect_tag': Y}
        'special_effects': {} # E.g., {'Revive': {'used': False, 'potency': Z}}
    }

    # Apply passive bonuses from equipment    
    for item_name, item_info in combatant['equipment'].items():
        if not isinstance(item_info, dict): continue # Safety check
        for effect, value in item_info.get('passive_effects', {}).items():
            if effect == 'flat_hp_bonus':
                combatant['hp'] += value
                combatant['max_hp'] += value
            # More robustly add to any matching base_stat key
            elif effect in combatant['base_stats']:
                 combatant['base_stats'][effect] += value            

        if 'on_hit_effect' in item_info:
            combatant['on_hit_effects'].append(item_info['on_hit_effect'])
        if 'on_being_hit_effect' in item_info:
            combatant['on_being_hit_effects'].append(item_info['on_being_hit_effect'])

        # Check for Revive specifically
        # Check both special_effect and on_being_hit_effect for Revive tag consistency
        special_tag = item_info.get('special_effect') # Old notebook structure check
        on_hit_tag = item_info.get('on_being_hit_effect', {}).get('effect_tag') # Your yaml structure check
        if special_tag == 'Revive' or on_hit_tag == 'Revive':
            if 'Revive' in effect_ruleset: # Get potency from ruleset
                 combatant['special_effects']['Revive'] = {
                     'used': False,
                     'potency': effect_ruleset['Revive'].get('potency', 0.3) # Default 30%
                 }            

    # Apply initial effects from room (matching notebook logic)    
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        if isinstance(room_rule, dict): # Ensure it's valid data
            # Apply immediate status effect if defined
            if 'effect_to_apply' in room_rule:
                effect_tag = room_rule['effect_to_apply']
                if effect_tag in effect_ruleset:
                    # Duration -1 indicates permanent
                    combatant['effects'][effect_tag] = {'duration': -1}                

            # Apply permanent stat modifier for the combat
            if 'stat_modifier' in room_rule:
                for stat_key, value in room_rule['stat_modifier'].items():
                    if stat_key in combatant['base_stats']:
                        combatant['base_stats'][stat_key] += value                    

    return combatant

def get_current_stats(combatant: Dict[str, Any], catalogs: Dict[str, Any]) -> Dict[str, float]:
    """Calculates current stats including active effects (matching notebook)."""
    
    current_stats = combatant['base_stats'].copy()
    effect_ruleset = catalogs.get('effects', {})

    for effect_name, effect_data in combatant.get('effects', {}).items():
        rule = effect_ruleset.get(effect_name, {})
        if isinstance(rule, dict) and 'stat_modifier' in rule:
            for stat_key, value in rule['stat_modifier'].items():
                if stat_key in current_stats:
                    # Apply additive modifiers
                    current_stats[stat_key] += value
                # Consider adding multiplicative modifiers if your ruleset uses them

    # Ensure critical stats don't go below reasonable limits
    current_stats['accuracy'] = max(0.0, current_stats.get('accuracy', 1.0))
    current_stats['evasion_chance'] = max(0.0, min(0.9, current_stats.get('evasion_chance', 0.0))) # Cap evasion
    current_stats['crit_chance'] = max(0.0, min(1.0, current_stats.get('crit_chance', 0.0)))
    current_stats['damage_reduction'] = max(0.0, min(0.9, current_stats.get('damage_reduction', 0.0))) # Cap reduction

    return current_stats

def resolve_turn_effects_and_cooldowns(combatant: Dict[str, Any], catalogs: Dict[str, Any], room_effect_name: Optional[str] = None) -> tuple[bool, bool]:
    """
    Applies start-of-turn effects (DoTs, HoTs, persistent room effects),
    decrements cooldowns, checks for death from effects, and handles Revive.
    Returns (is_stunned, died_from_effects). Matches notebook logic flow.
    """
    
    effect_ruleset = catalogs.get('effects', {})
    room_effect_catalog = catalogs.get('room_effects', {})
    stats = get_current_stats(combatant, catalogs) # Get stats *before* applying dots/hots

    is_stunned = False
    died_from_effects = False
    hp_before_effects = combatant['hp']

    # Apply persistent effects from room first (e.g., Consecrated Ground heal)    
    if room_effect_name and room_effect_name in room_effect_catalog:
        room_rule = room_effect_catalog[room_effect_name]
        if isinstance(room_rule, dict) and 'persistent_effect' in room_rule:
            persistent_rule = room_rule['persistent_effect']
            if persistent_rule.get('type') == 'Heal_Self':
                heal_per_round = persistent_rule.get('heal_per_round', 0.0) # Expects percentage
                if heal_per_round > 0:
                    heal_amount = combatant['max_hp'] * heal_per_round
                    combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)

    # Apply active status effects (DoTs, HoTs) and check for stun    
    active_effects = list(combatant.get('effects', {}).keys()) # Iterate over copy
    for effect_name in active_effects:
        if effect_name not in combatant['effects']: continue # Check if removed during iteration (unlikely here)

        rule = effect_ruleset.get(effect_name, {})
        if not isinstance(rule, dict): continue

        if rule.get('type') == 'DoT':
            # Use dot_potency_modifier from current stats
            dot_damage = rule.get('damage_per_turn', 0) * stats.get('dot_potency_modifier', 1.0)
            if dot_damage > 0:
                combatant['hp'] -= dot_damage
        elif rule.get('type') == 'Heal_Self': # HoT based on potency (percentage of max HP)
            potency = rule.get('potency', 0.0)
            if potency > 0:
                heal_amount = combatant['max_hp'] * potency
                combatant['hp'] = min(combatant['max_hp'], combatant['hp'] + heal_amount)
            # Add flat_amount HoT if your ruleset uses it

        # Check for stun effects
        if rule.get('type') == 'Control' and effect_name in ['Stun', 'Fear']: # Fear also stuns in notebook
            is_stunned = True

        # Decrement duration for non-permanent effects
        duration = combatant['effects'][effect_name].get('duration', 0)
        if duration > 0: # Only decrement positive durations
            combatant['effects'][effect_name]['duration'] -= 1
            if combatant['effects'][effect_name]['duration'] <= 0:
                del combatant['effects'][effect_name]
        # duration == -1 means permanent (from room effect or special skill)
        # duration == 0 means instant effect (shouldn't really be in 'effects' dict long)

    # Check for death after DoTs and handle Revive (Notebook logic integrated)    
    if combatant['hp'] <= 0:
        revive_status = combatant.get('special_effects', {}).get('Revive')
        if revive_status and not revive_status.get('used', True): # Check used is explicitly False
            revive_potency = revive_status.get('potency', 0.0)
            combatant['hp'] = combatant['max_hp'] * revive_potency
            combatant['special_effects']['Revive']['used'] = True
            is_stunned = False # Revived combatant can act
            died_from_effects = False # They didn't *stay* dead            
        else:
            died_from_effects = True
            is_stunned = True # Can't act if dead

    # Decrement cooldowns (Done at start of turn/step)
    # - moved from end of action phase
    if not died_from_effects: # Don't decrement if dead
        for skill_name in combatant.get('cooldowns', {}):
            if combatant['cooldowns'][skill_name] > 0:
                combatant['cooldowns'][skill_name] -= 1

    return is_stunned, died_from_effects


def execute_action(
    attacker: Dict[str, Any],
    targets: List[Dict[str, Any]], # Assumes non-empty list of potential targets
    action_name: str,
    catalogs: Dict[str, Any]
) -> None: # Returns None, modifies combatants directly
    """
    Executes a single action (skill) based on notebook logic.
    Calculates hit, damage/heal, applies effects (on-hit, skill tags).
    Sets cooldown. Does NOT check for death here.
    """
    #
    if action_name == 'Wait' or not targets:
        return

    skill_catalog = catalogs.get('skills', {})
    effect_ruleset = catalogs.get('effects', {})
    
    if action_name not in attacker.get('skills', {}):          
        return
    if attacker.get('cooldowns', {}).get(action_name, 0) > 0:        
        # In a real game, this might be an invalid action penalized by env.py
        return 

    skill_info = attacker['skills'][action_name]
    attacker_stats = get_current_stats(attacker, catalogs)

    # --- Target Selection (from notebook) ---
    is_aoe = 'AOE' in skill_info.get('tags', []) #
    
    # Filter targets to living opponents (or allies for heals/buffs)
    skill_target_type = skill_info.get('target', 'Target') # Default to enemy
    
    possible_targets = []
    if skill_target_type in ['Target', 'AOE']: # Enemy targeting skills
         possible_targets = [t for t in targets if isinstance(t, dict) and t.get('hp', 0) > 0 and t.get('team') != attacker.get('team')]
    elif skill_target_type == 'Self': # Self targeting skills
         possible_targets = [attacker]
    # Add 'Ally' targeting if needed
         
    if not possible_targets:
        # print(f"Debug: {attacker['name']} used {action_name} but no valid targets found.")
        # Set cooldown even if no target found? Yes, skill was attempted.
        attacker['cooldowns'][action_name] = skill_info.get('cd', 0)
        return

    # Select final target(s)
    final_targets = []
    if is_aoe:
        final_targets = possible_targets # Hit all valid targets
    elif skill_target_type == 'Self':
        final_targets = [attacker]
    else: # Single target enemy/ally
        final_targets = [random.choice(possible_targets)] # Random choice among valid targets

    # Action Resolution Loop (Apply to each final target)
    for target in final_targets:
        if not isinstance(target, dict) or target.get('hp', 0) <= 0: continue # Double check target validity

        target_stats = get_current_stats(target, catalogs)
        base_damage = float(skill_info.get('damage', 0)) # Ensure float

        # Damage/Healing Calculation
        final_damage_taken = 0.0 # Track for Reflect
        
        if base_damage > 0: # DAMAGING SKILL
            # Hit Chance (ensure bounds 5%-95% like notebook)
            hit_chance = attacker_stats.get('accuracy', 1.0) - target_stats.get('evasion_chance', 0.0)
            hit_chance = max(0.05, min(hit_chance, 0.95))

            if random.random() < hit_chance: # IT HITS
                # Modifiers
                damage_mod = attacker_stats.get('damage_modifier', 1.0)
                # Use defense_taken stat for vulnerability checks
                defense_mod = (1.0 - target_stats.get('damage_reduction', 0.0)) * target_stats.get('defense_taken', 1.0)
                defense_mod = max(0.1, defense_mod) # Ensure minimum damage passage
                
                is_crit = random.random() < attacker_stats.get('crit_chance', 0.0)
                crit_modifier = 2.0 if is_crit else 1.0
                
                # Damage Formula
                final_damage = (base_damage + attacker_stats.get('flat_damage_bonus', 0.0)) * damage_mod * crit_modifier
                final_damage_taken = max(0.0, final_damage * defense_mod) # Damage cannot be negative

                # Apply Damage
                target['hp'] -= final_damage_taken

                # Apply Attacker's ON-HIT effects (to Target)                
                for on_hit in attacker.get('on_hit_effects', []):
                    if random.random() < on_hit.get('chance', 0.0):
                        tag = on_hit.get('effect_tag')
                        if tag and tag in effect_ruleset:
                            rule = effect_ruleset[tag]
                            if 'duration' in rule: # Apply only if it has duration
                                target['effects'][tag] = {'duration': rule['duration']}

                # Apply Target's ON-BEING-HIT effects (can affect Attacker)                
                if target['hp'] > 0: # Only if target survives
                    for on_being_hit in target.get('on_being_hit_effects', []):
                        if random.random() < on_being_hit.get('chance', 0.0):
                            tag = on_being_hit.get('effect_tag')
                            if tag and tag in effect_ruleset:
                                rule = effect_ruleset[tag]
                                if tag == 'Reflect': #
                                    reflected_damage = final_damage_taken * rule.get('potency', 0.0)
                                    attacker['hp'] -= reflected_damage
                                elif tag == 'Revive': # Handled separately
                                    pass
                                elif 'duration' in rule: # Apply effect (e.g., Fear) to Attacker
                                    attacker['effects'][tag] = {'duration': rule['duration']}
            # else: Missed!

        elif base_damage < 0: # HEALING SKILL (Negative damage value)
            heal_amount = abs(base_damage)
            # Potentially add heal modifiers based on stats if needed
            target['hp'] = min(target['max_hp'], target['hp'] + heal_amount)        

    # Apply SKILL TAGS (Buffs/Debuffs etc.) - Applied ONCE per skill use
    # Moved from inside the target loop and hit check
    for tag in skill_info.get('tags', []):
        if tag in effect_ruleset:
            rule = effect_ruleset[tag]
            if isinstance(rule, dict) and 'duration' in rule: # Check it's a valid rule with duration
                
                effect_target_type = rule.get('target', 'Target') # Default to enemy/target
                
                # Determine targets for this specific effect tag
                if effect_target_type == 'Self':
                    effect_apply_targets = [attacker]
                elif effect_target_type == 'Target': # Apply to all targets hit/affected by the skill
                    effect_apply_targets = final_targets
                # Add 'Ally' if needed
                else:
                    effect_apply_targets = final_targets # Default case

                # Apply the effect tag to the determined targets
                for et in effect_apply_targets:
                     if isinstance(et, dict) and et.get('hp', 0) > 0: # Check if target is valid and alive
                         # Apply or refresh duration
                         et['effects'][tag] = {'duration': rule['duration']}
            # else: tag might be informational (like Physical) or instant (handled by damage/heal)

    # Set Cooldown (Done ONCE after the skill completes)    
    attacker['cooldowns'][action_name] = skill_info.get('cd', 0)

def check_for_death_and_revive(combatant: Dict[str, Any], catalogs: Dict[str, Any]) -> bool:
    """
    Checks if HP <= 0 and handles Revive if available.
    Returns True if permanently dead, False otherwise. Matches notebook logic.
    """
    #
    if combatant.get('hp', 0) <= 0:
        revive_status = combatant.get('special_effects', {}).get('Revive')
        if revive_status and not revive_status.get('used', True): # Check used is explicitly False
            revive_potency = revive_status.get('potency', 0.0)
            combatant['hp'] = combatant['max_hp'] * revive_potency
            combatant['special_effects']['Revive']['used'] = True            
            return False # Not permanently dead
        else:
            return True # Permanently dead
    return False # Still alive