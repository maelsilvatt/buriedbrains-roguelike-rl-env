/**
 * asset_loader.js
 */
window.AssetManager = {
    paths: {
        agents: '../../assets/agent/',
        enemies: '../../assets/enemies/' 
    },
        
    agentAssets: [
        { sprite: 'brute-female.png',     portrait: 'brute-female-portrait.png' },
        { sprite: 'brute-male.png',       portrait: 'brute-male-portrait.png' },

        { sprite: 'experient-female.png', portrait: 'experient-female-portrait.png' },
        { sprite: 'experient-male.png',   portrait: 'experient-male-portrait.png' },

        { sprite: 'mage-female.png',      portrait: 'mage-female-portrait.png' },
        { sprite: 'mage-male.png',        portrait: 'mage-male-portrait.png' },

        { sprite: 'explorer-female.png',  portrait: 'explorer-female-portrait.png' },
        { sprite: 'explorer-male.png',    portrait: 'explorer-male-portrait.png' },

        { sprite: 'fanatic-female.png',   portrait: 'fanatic-female-portrait.png' },
        { sprite: 'fanatic-male.png',     portrait: 'fanatic-male-portrait.png' },

        { sprite: 'hopeless-female.png',  portrait: 'hopeless-female-portrait.png' },
        { sprite: 'hopeless-male.png',    portrait: 'hopeless-male-portrait.png' },

        { sprite: 'young-female.png',     portrait: 'young-female-portrait.png' },
        { sprite: 'young-male.png',       portrait: 'young-male-portrait.png' },
    ],

    enemyMap: {        
        'Ashen Monarch': 'ashen-monarch.png',
        'A Thousand Cursed': 'a-thousand-cursed.png',
        'Bandit': 'bandit.png',
        'Bedrose Gardener': 'bedrose-gardener.png',
        'Berserker Spirit': 'berserker-spirit.png',
        'Cave Spider': 'cave-spider.png',
        'Cursed Knight': 'cursed-knight.png',
        'Cursed Spirit': 'cursed-spirit.png',
        'Cursed Zombie': 'cursed-zombie.png',
        'Dark Cultist': 'dark-cultist.png',
        'Flesh Abomination': 'flesh-abomination.png',
        'Forest Wisp': 'forest-wisp.png',
        'Giant Bat': 'giant-bat.png',        
        'Goblin': 'goblin.png',
        'Hopeless Agent': 'hopeless-agent.png',
        'Kalisto': 'kalisto.png',
        'Krul': 'krul.png',
        'Laezarus': 'laezarus.png',
        'Lava Golem': 'lava-golem.png',
        'Lich': 'lich.png',
        'Poisoned Rat': 'poisoned-rat.png',
        'Poisoned Scorpion': 'poisoned-scorpion.png',
        'Poisonous Mushroom': 'poisonous-mushroom.png',
        'Queen Arachna': 'queen-arachna.png',
        'Shadow Assassin': 'shadow-assassin.png',
        'Shadow Lurker': 'shadow-lurker.png',        
        'Skeleton Archer': 'skeleton-archer.png',
        'The Dark Knight': 'the-dark-knight.png',
        'The Forgotten One': 'the-forgotten-one.png',
        'Twilight Stalker': 'twilight-stalker.png',
        'Wild Boar': 'wild-boar.png',
        'Zombie': 'zombie.png'
    },
    
    defaults: {
        sprite: 'explorer-male.png', 
        portrait: 'explorer-male-portrait.png',
        enemy: 'bandit.png'
    },

    cache: {},

    preloadAll: function(onProgress) {
        const allAssets = [];

        this.agentAssets.forEach(asset => {
            if (asset.sprite) allAssets.push(this.paths.agents + asset.sprite);
            if (asset.portrait) allAssets.push(this.paths.agents + asset.portrait);
        });

        Object.values(this.enemyMap).forEach(file => {
            allAssets.push(this.paths.enemies + file);
        });

        console.log(`🔄 Carregando ${allAssets.length} assets...`);

        let loaded = 0;
        const updateProgress = () => {
            loaded++;
            if (onProgress) onProgress(loaded, allAssets.length);
        };

        const promises = allAssets.map(src => this._loadImage(src).then(updateProgress));
        return Promise.all(promises);
    },

    getAgentSprite(agentName) {
        const asset = this._getAgentAssetPair(agentName);
        return this.paths.agents + (asset.sprite || this.defaults.sprite);
    },

    getAgentPortrait(agentName) {
        const asset = this._getAgentAssetPair(agentName);
        return this.paths.agents + (asset.portrait || this.defaults.portrait);
    },

    getEnemyImage(enemyName) {
        if (!enemyName) return this.paths.enemies + this.defaults.enemy;
        if (this.enemyMap[enemyName]) return this.paths.enemies + this.enemyMap[enemyName];

        const lowerName = enemyName.toLowerCase();
        for (const key in this.enemyMap) {
            if (lowerName.includes(key.toLowerCase())) {
                return this.paths.enemies + this.enemyMap[key];
            }
        }
        return this.paths.enemies + this.defaults.enemy;
    },

    isLoaded(src) {
        return !!this.cache[src];
    },

    _loadImage(src) {
        return new Promise((resolve) => {
            if (this.cache[src]) return resolve(src); // já carregado

            const img = new Image();
            img.src = src;
            img.onload = () => {
                this.cache[src] = img;
                resolve(src);
            };
            img.onerror = () => {
                console.warn(`❌ Erro ao carregar: ${src}`);
                resolve(null);
            };
        });
    },

    _getAgentAssetPair(agentName) {
        if (!agentName) return this.defaults;
        const index = this._stringToHash(agentName) % this.agentAssets.length;
        return this.agentAssets[index];
    },

    _stringToHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
};