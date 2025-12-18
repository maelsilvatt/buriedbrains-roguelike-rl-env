import sys
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    print("Erro: A biblioteca 'ruamel.yaml' é necessária para não apagar seus comentários.")
    print("Instale com: pip install ruamel.yaml")
    sys.exit(1)

# CONFIGURAÇÃO DE BALANCEAMENTO (CALIBRADA)
# Estes valores foram ajustados para caber na curva de budget:
# Budget Base = 100 + (Andar * 10)
ECONOMY_CONFIG = {
    # Pesos de raridade para multiplicar o custo final
    "rarity_mult": {
        # Equipamentos
        "Common": 1.0,
        "Rare": 1.5,      # Aprox. Andar 10-20
        "Epic": 2.2,      # Aprox. Andar 30-50
        "Legendary": 3.5, # Aprox. Andar 80+
        
        # Inimigos (Budget de spawn)
        "Fodder": 0.5,     
        "Common": 1.0,
        "Professor": 1.5,
        "Tank": 1.8,
        "Elite": 2.5,      
        "Boss": 6.0       # Bosses são caros para garantir que venham sozinhos ou em salas especiais
    },

    # Pesos para stats individuais    
    "stat_weights": {
        # Atributos Planos
        "flat_damage_bonus": 0.5,    # 2 de Dano = 1 Gold
        "flat_hp_bonus": 0.1,        # 10 HP = 1 Gold
        "base_damage": 1.0,          # Dano base de inimigo vale mais (é mais perigoso)
        
        # Modificadores Percentuais (Valores no YAML são 0.1, 0.5 etc)
        # Ex: 0.1 (10%) * 80 = 8 de custo adicionado
        "damage_modifier": 80.0,     
        "damage_reduction": 100.0,   
        "speed": 50.0,
        "evasion_chance": 60.0,
        "defense": 70.0,            
    },

    # Custos base por tipo de item
    "base_costs": {
        "Weapon": 15,
        "Armor": 10,
        "Artifact": 20,   # Valorizado pois é token social na Arena
        "SkillTome": 10,  # Caro, pois define a build e dá muito reward
        "Enemy": 5        # Custo mínimo para existir
    },

    # Pesos de efeitos especiais
    "effect_weights": {
        "Stun": 20,
        "Poison": 12,
        "Burn": 12,
        "Vulnerable": 15,
        "Reflect": 25,
        "Revive": 150,    # Phoenix Feather (Item de "vida extra")
        "Sunder": 15,
        "Stealth": 10,
    }
}

def calculate_item_cost(item_name, data):
    """Calcula o custo de Equipamentos, Artefatos e Grimórios."""
    if not isinstance(data, dict):
        return None

    item_type = data.get('type', 'Unknown')
    rarity = data.get('rarity', 'Common')
    
    # Valor Base
    total_score = ECONOMY_CONFIG["base_costs"].get(item_type, 10)

    # Somar passivas (flat e percentuais)
    passives = data.get('passive_effects', {}) or {}
    for stat, value in passives.items():
        weight = ECONOMY_CONFIG["stat_weights"].get(stat, 0)
        total_score += value * weight

    # Analisar Efeitos On-Hit/Being-Hit
    for effect_key in ['on_hit_effect', 'on_being_hit_effect']:
        effect_data = data.get(effect_key)
        if effect_data:
            tag = effect_data.get('effect_tag')
            chance = effect_data.get('chance', 1.0)
            effect_val = ECONOMY_CONFIG["effect_weights"].get(tag, 10)
            total_score += effect_val * chance

    # SkillTomes    
    # Aplica multiplicador de raridade
    multiplier = ECONOMY_CONFIG["rarity_mult"].get(rarity, 1.0)
    final_cost = total_score * multiplier

    return int(round(final_cost))

def calculate_enemy_cost(enemy_name, data):
    """Calcula o custo (budget) de Inimigos."""
    if not isinstance(data, dict):
        return None
    
    tier = data.get('tier', 'Common')
    
    # Custo Base
    score = ECONOMY_CONFIG["base_costs"]["Enemy"]

    # Stats Base do Inimigo
    stats = data.get('base_stats', {}) or {}
    score += stats.get('damage', 0) * ECONOMY_CONFIG["stat_weights"]["base_damage"]
    score += stats.get('defense', 0) * ECONOMY_CONFIG["stat_weights"]["defense"]
    
    # Tags também podem influenciar se desejar
    tags = data.get('tags', [])
    for tag in tags:
        # Ex: Stealth aumenta o custo levemente
        if tag in ECONOMY_CONFIG["effect_weights"]:
             score += ECONOMY_CONFIG["effect_weights"][tag]

    # Multiplicador de Tier (Fodder, Elite, Boss)
    multiplier = ECONOMY_CONFIG["rarity_mult"].get(tier, 1.0)
    
    final_cost = score * multiplier
    
    # Regra de segurança: Bosses nunca custam menos que 20, Fodder nunca menos que 2
    return int(max(2, round(final_cost)))

def process_file(file_path, processor_type):
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    print(f"🔄 Processando {file_path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    updated_count = 0
    
    if processor_type == 'equipment':
        # Procura a chave raiz (ex: equipment_catalog) ou usa o dict inteiro
        root = data.get('equipment_catalog', data)
        
        for item_name, item_data in root.items():
            if isinstance(item_data, dict):
                new_cost = calculate_item_cost(item_name, item_data)
                if new_cost is not None:
                    item_data['cost'] = new_cost
                    updated_count += 1
                    
    elif processor_type == 'enemies':
        # Estrutura: pools -> enemies
        pools = data.get('pools', {})
        enemies = pools.get('enemies', {})
        
        for name, enemy_data in enemies.items():
            new_cost = calculate_enemy_cost(name, enemy_data)
            if new_cost is not None:
                enemy_data['cost'] = new_cost
                updated_count += 1

    # Salva Backup (.bak)
    backup_path = path.with_suffix('.yaml.bak')
    with open(backup_path, 'w', encoding='utf-8') as f_bak:
        yaml.dump(data, f_bak)
    print(f"   ↳ Backup criado em {backup_path}")
        
    # Salva Original Atualizado
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
        
    print(f"✅ Concluído: {updated_count} itens atualizados em {file_path}.\n")

if __name__ == "__main__":
    print("Economia de Buried Brains - Automação de Custo de Itens e Inimigos")
    print("Regra de Budget alvo: 100 + (Andar * 10)\n")
    
    # Processa Equipamentos
    process_file("buriedbrains/data/equipment_catalog.yaml", "equipment")
    
    # Processa Inimigos
    process_file("buriedbrains/data/enemies_and_events.yaml", "enemies")
    
    print("Automação finalizada.")