// 1. INICIALIZAÇÃO E DADOS
const RECORD_PATH = '../recordings/sanctum_floor_25_expert_recording.json';
let replayData = []; 
let currentFrame = 0;
let isPlaying = false;
let playInterval;

// Mock de segurança caso não ache o arquivo
const mockReplay = [
    // FRAME 0: Início
    {
        turn: 1,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0, imag: 0},
                location_node: "Start", floor: 1, scene_mode: "EXPLORATION",
                logs: ["Início da Jornada."],
                neighbors: [
                    {id: "p_1_0", has_enemy: true, enemy_type: "Goblin Scout", has_treasure: false},
                    {id: "p_1_1", has_enemy: false, has_treasure: true},
                    {id: "p_1_2", has_enemy: false, has_treasure: false, effect: "Heat Wave"}
                ]
            }
        }
    },
    // FRAME 1: Encontro com Inimigo
    {
        turn: 2,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0, imag: 0},
                location_node: "p_1_0", floor: 1, scene_mode: "COMBAT_PVE", // <--- Muda para combate
                combat_enemy: "Goblin Scout",
                logs: ["Entrou na sala p_1_0.", "Encontrou Goblin Scout!"],
                neighbors: []
            }
        }
    },
    // FRAME 2: Tomando Dano (Para testar o Shake)
    {
        turn: 3,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 90, max_hp: 100, level: 1, karma: {real: -0.1, imag: 0.1},
                location_node: "p_1_0", floor: 1, scene_mode: "COMBAT_PVE",
                combat_enemy: "Goblin Scout",
                logs: ["Sofreu 10 de dano."],
                neighbors: []
            }
        }
    },
    // FRAME 3: Vitória e Drop
    {
        turn: 4,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 90, max_hp: 100, level: 2, karma: {real: -0.1, imag: 0.1}, // Level UP
                location_node: "p_1_0", floor: 1, scene_mode: "EXPLORATION",
                logs: ["Inimigo derrotado!", "Level Up!", "Item encontrado: Iron Sword"],
                neighbors: [
                    {id: "p_2_0", has_enemy: true, enemy_type: "Elite Knight (X)", has_treasure: true} // Teste do Elite
                ]
            }
        }
    }
];

// Efeitos de Sala 
const EFFECT_LIBRARY = {
    // --- Negativos ---
    'Slow Terrain': { 
        desc: "O terreno difícil aplica 'Slow' em todos no início do combate.", 
        type: 'debuff', icon: 'fa-shoe-prints' 
    },
    'Heat': { 
        desc: "O calor intenso aplica 'Burn' em todos a cada rodada.", 
        type: 'debuff', icon: 'fa-fire' 
    },
    'Dense Fog': { 
        desc: "A neblina densa aplica 'Blind' em todos no início do combate.", 
        type: 'debuff', icon: 'fa-smog' 
    },
    'Weakening Field': { 
        desc: "Uma aura profana reduz a eficácia de curas em 50% na sala.", 
        type: 'debuff', icon: 'fa-heart-crack' 
    },

    // --- Positivos ---
    'Evasion Zone': { 
        desc: "Uma névoa mágica concede bônus de evasão a todos.", 
        type: 'buff', icon: 'fa-wind' 
    },
    'Amplifying Field': { 
        desc: "O campo de energia aumenta o dano de todos em 15%.", 
        type: 'buff', icon: 'fa-hand-fist' 
    },
    'Consecrated Ground': { 
        desc: "O solo sagrado cura todos os combatentes em 2% HP a cada rodada.", 
        type: 'buff', icon: 'fa-notes-medical' 
    },
    'Conductive Field': { 
        desc: "A energia arcana aumenta dano mágico causado e recebido em 25%.", 
        type: 'neutral', icon: 'fa-bolt' 
    }
};

// 2. RENDERIZAÇÃO

function renderFrame(index) {
    if (!replayData || index >= replayData.length) { pause(); return; }
    
    // Suporte a múltiplos agentes (pega o primeiro encontrado se 'a1' não existir)
    const frame = replayData[index];
    const agentId = Object.keys(frame.agents)[0]; 
    const agent = frame.agents[agentId];

    // Atualiza UI
    document.getElementById('floor-display').innerText = `FLOOR ${agent.floor}`;
    document.getElementById('turn-display').innerText = frame.turn;
    document.getElementById('timeline').value = index;
    
    updateStats(agent);
    drawKarma(agent.karma);
    updateLogs(agent.logs);    
    updateStats(agent, agent.action_taken);

    const stage = document.getElementById('stage');
    if (agent.scene_mode.includes("COMBAT")) {
        renderCombatScene(stage, agent, index);
    } else {        
        renderExplorationScene(stage, agent, index); 
    }
}

function updateStats(agent) {
    // --- BARRAS (HP / XP) ---
    const hpPct = Math.max(0, (agent.hp / agent.max_hp) * 100);
    const xpPct = agent.exp_percent || 0;
    
    document.getElementById('hp-bar').style.width = `${hpPct}%`;
    document.getElementById('hp-text').innerText = `${Math.ceil(agent.hp)}/${agent.max_hp}`;
    
    document.getElementById('xp-bar').style.width = `${xpPct}%`;
    document.getElementById('xp-text').innerText = `XP ${xpPct}%`;
    
    // Atualiza Avatar (opcional)
    const avatar = document.getElementById('agent-avatar');
    if(agent.hp <= 0) avatar.style.filter = "grayscale(100%)";
    else avatar.style.filter = "none";

    // --- EQUIPAMENTOS ---
    updateEquipmentSlots(agent.equipment);

    // --- SKILLS ---
    let actionName = null;
    if (agent.action_taken !== undefined && agent.action_taken !== null) {
        if (typeof agent.action_taken === 'number') {
            actionName = ACTION_INDEX_MAP[agent.action_taken];
        } else {
            actionName = agent.action_taken; 
        }
    }

    updateEquipmentSlots(agent.equipment);
    
    if (agent.cooldowns) {
        updateSkills(agent.cooldowns, actionName);
    }
}

// Mapeamento de Skills e seus CDs Máximos (Para forçar visualização)
const SKILL_INFO = {
"Quick Strike": { id: "skill-0", maxCD: 1 },
"Heavy Blow":   { id: "skill-1", maxCD: 3 },
"Stone Shield": { id: "skill-2", maxCD: 3 },
"Wait":         { id: "skill-3", maxCD: 0 }
};

// Mapeamento de índices de ação (do Env) para nomes
// 0: Quick Strike, 1: Heavy Blow, 2: Stone Shield, 3: Wait
const ACTION_INDEX_MAP = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"];

// --- FUNÇÃO DE ATUALIZAÇÃO DE SKILLS ---
function updateSkills(cooldowns, currentAction) {
    Object.keys(SKILL_INFO).forEach(skillName => {
        const info = SKILL_INFO[skillName];
        const element = document.getElementById(info.id);
        if(!element) return;

        const overlay = element.querySelector('.cd-overlay');
        
        // Valor do JSON
        let cdValue = cooldowns[skillName] || 0;
        
        // Força a visualização como se já tivesse apertado o botão
        if (currentAction === skillName && info.maxCD > 0) {
            cdValue = info.maxCD;
        }

        if (cdValue > 0) {
            element.classList.add('on-cooldown');
            if(overlay) {
                overlay.style.opacity = '1';
                overlay.innerText = cdValue;
            }
        } else {
            element.classList.remove('on-cooldown');
            if(overlay) {
                overlay.style.opacity = '0';
                overlay.innerText = "";
            }
        }
    });
}

function drawKarma(karma) {
    const canvas = document.getElementById('karma-disk');
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, r = w/2;
    
    ctx.clearRect(0,0,w,h);
    
    // Disco Fundo
    ctx.beginPath();
    ctx.arc(r, r, r-2, 0, 2*Math.PI);
    ctx.fillStyle = '#222'; ctx.strokeStyle = '#555';
    ctx.fill(); ctx.stroke();
    
    // Ponto
    const kx = r + (karma.real * (r-8));
    const ky = r - (karma.imag * (r-8)); // Y invertido matemático
    
    ctx.beginPath();
    ctx.arc(kx, ky, 5, 0, 2*Math.PI);
    
    // Cor baseada na posição (Santo=Verde, Vilão=Vermelho)
    let color = '#ffeaa7'; // Neutro
    if (karma.real > 0.3) color = '#55efc4';
    if (karma.real < -0.3) color = '#ff7675';
    
    ctx.fillStyle = color;
    ctx.fill();
}

function updateLogs(logs) {
    const container = document.getElementById('log-container');
    container.innerHTML = logs.map(l => `<div class="log-entry">> ${l}</div>`).join('');
    container.scrollTop = container.scrollHeight;
}

// --- CENA EXPLORAÇÃO VERTICAL ---
function renderExplorationScene(container, agent, index) { 
    container.innerHTML = '';
    
    // 1. Título da Sala Atual
    const titleDiv = document.createElement('div');
    titleDiv.className = 'room-title-display';
    titleDiv.innerText = `CURRENT: ${formatRoomName(agent.location_node)}`;
    container.appendChild(titleDiv);

    const col = document.createElement('div');
    col.className = 'exploration-container';

    // --- PREVISÃO DO FUTURO (Qual sala ele escolheu?) ---
    let chosenNodeId = null;
    // Verifica se existe um próximo frame
    if (index < replayData.length - 1) {
        const nextFrame = replayData[index + 1];
        // Assume agente 'a1' ou pega dinâmico
        const nextAgent = nextFrame.agents[Object.keys(nextFrame.agents)[0]];
        chosenNodeId = nextAgent.location_node;
    }

    // Tier 1: Oculto (Futuro)
    const row1 = document.createElement('div');
    row1.className = 'row-tier';
    for(let i=0; i<4; i++) row1.appendChild(createCard('UNKNOWN', [], 'hidden'));
    col.appendChild(row1);

    // Tier 2: Vizinhos (Opções)
    const row2 = document.createElement('div');
    row2.className = 'row-tier';
    
    if (agent.neighbors && agent.neighbors.length > 0) {
        agent.neighbors.forEach(n => {
            let icons = [];
            
            // Ícones (Mesma lógica de antes)
            if (n.has_enemy) {
                const rank = detectEnemyRank(n.enemy_type);
                icons.push({ class: 'fa-skull', color: `skull-${rank}`, tip: `Inimigo: ${n.enemy_type}` });
            }
            if (n.has_treasure) icons.push({class: 'fa-gem', color: 'icon-event', tip: 'Tesouro'});
            if (n.effect && n.effect !== 'None') icons.push({class: 'fa-wind', color: 'icon-effect', tip: n.effect});
            if (n.is_exit) icons.push({class: 'fa-door-open', color: 'icon-door', tip: 'Saída'});
            if (icons.length === 0 && !n.is_exit) icons.push({class: 'fa-road', color: 'icon-unknown', tip: 'Vazio'});

            // CRIA O CARD
            const card = createCard(formatRoomName(n.id), icons, 'choice');

            // --- LÓGICA DE DESTAQUE ---
            if (n.id === chosenNodeId) {
                card.classList.add('selected'); // Borda branca e Glow
                
                // Adiciona o indicador em baixo
                const indicator = document.createElement('div');
                indicator.className = 'selection-indicator';
                indicator.innerText = "CHOSEN"; // Ou "ESCOLHIDO"
                card.appendChild(indicator);
            }

            row2.appendChild(card);
        });
    } else {
        row2.appendChild(createCard('WALL', [{class: 'fa-ban', color: 'icon-unknown'}], 'wall'));
    }
    col.appendChild(row2);

    // Tier 3: Agente (Presente)
    const row3 = document.createElement('div');
    row3.className = 'row-tier';
    const agentIcons = [{class: 'fa-user-ninja', color: 'icon-agent', tip: 'Você'}];
    const agCard = createCard('HERO', agentIcons, 'self');
    agCard.classList.add('card-agent');
    row3.appendChild(agCard);
    
    col.appendChild(row3);
    container.appendChild(col);
}

// --- FUNÇÃO DE EQUIPAMENTO ---
function updateEquipmentSlots(equipmentList) {    
    const slots = {
        weapon: document.getElementById('slot-weapon'),
        armor: document.getElementById('slot-armor'),
        artifact: document.getElementById('slot-artifact')
    };

    // Reseta classes
    resetSlot(slots.weapon, 'fa-fist-raised');
    resetSlot(slots.armor, 'fa-shirt');
    resetSlot(slots.artifact, 'fa-ring');

    if (!equipmentList) return;

    equipmentList.forEach(item => {
        if(!item) return;
        const name = item.toLowerCase();
        
        let targetSlot = null;
        let iconClass = '';

        // Detecção do Tipo de Item
        if (name.includes('sword') || name.includes('bow') || name.includes('axe') || name.includes('staff') || name.includes('dagger') || name.includes('spear')) {
            targetSlot = slots.weapon;
            iconClass = 'fa-khanda'; // Ícone de espada
        } else if (name.includes('armor') || name.includes('plate') || name.includes('robe') || name.includes('chain') || name.includes('tunic')) {
            targetSlot = slots.armor;
            iconClass = 'fa-shield-halved';
        } else if (name.includes('amulet') || name.includes('ring') || name.includes('charm') || name.includes('idol')) {
            targetSlot = slots.artifact;
            iconClass = 'fa-gem';
        }

        // Se encontrou onde colocar
        if (targetSlot) {
            const rarity = detectRarity(name);
            
            // Remove bordas antigas
            targetSlot.className = `equip-slot equipped border-${rarity}`;
            targetSlot.title = item; // Tooltip com nome
            targetSlot.innerHTML = `<i class="fa-solid ${iconClass}"></i>`;
        }
    });
}

function resetSlot(el, defaultIcon) {
    el.className = 'equip-slot border-common'; // Volta pro padrão
    el.innerHTML = `<i class="fa-solid ${defaultIcon}"></i>`;
    el.title = "Vazio";
    el.classList.remove('equipped');
}

// --- LÓGICA DE RARIDADE ---
function detectRarity(itemName) {
    const n = itemName.toLowerCase();
    
    // Lendário (Laranja)
    if (n.includes('legendary') || n.includes('godly') || n.includes('ancient') || n.includes('dragon')) return 'legendary';
    
    // Épico (Roxo)
    if (n.includes('epic') || n.includes('obsidian') || n.includes('crystal') || n.includes('shadow')) return 'epic';
    
    // Raro (Azul)
    if (n.includes('rare') || n.includes('steel') || n.includes('silver') || n.includes('reinforced')) return 'rare';
    
    // Comum (Verde - Padrão)
    return 'common';
}

// --- CENA COMBATE ---
function renderCombatScene(container, agent, index) {
    container.innerHTML = '';
    
    // 1. Lógica de Dano e Comparação com Frame Anterior
    const prevFrame = index > 0 ? replayData[index - 1] : null;
    let prevAgent = null;
    let wasInCombatBefore = false;

    if (prevFrame) {
        const prevAgentId = Object.keys(prevFrame.agents)[0];
        prevAgent = prevFrame.agents[prevAgentId];
        // Verifica se no frame passado JÁ ESTAVA em combate PVE
        if (prevAgent.scene_mode && prevAgent.scene_mode.includes("COMBAT")) {
            wasInCombatBefore = true;
        }
    }

    // Shake no Heroi
    const heroHurt = prevAgent && prevAgent.hp > agent.hp ? 'shake' : '';
    
    // 2. Lógica do Inimigo
    let enemyHurt = '';
    let enemyHpPct = 100;
    let enemyName = "Inimigo";
    let enemyHpText = "??/??";

    if (agent.combat_data) {
        enemyName = agent.combat_data.name;
        const curHp = agent.combat_data.hp;
        const maxHp = agent.combat_data.max_hp;
        enemyHpPct = Math.max(0, (curHp / maxHp) * 100);
        enemyHpText = `${Math.ceil(curHp)}/${maxHp}`;

        if (prevAgent && prevAgent.combat_data && prevAgent.combat_data.hp > curHp) {
            enemyHurt = 'shake';
        }
    }
    
    // 3. Lógica do Badge de Efeito (Com controle de animação)
    let effectHtml = '';
    const effectName = agent.current_effect; 
    
    if (effectName && effectName !== 'None' && EFFECT_LIBRARY[effectName]) {
        const data = EFFECT_LIBRARY[effectName];
        
        // Só anima se NÃO estava em combate antes (Entrada)
        const animClass = wasInCombatBefore ? '' : 'anim-enter';
        
        effectHtml = `
            <div class="combat-effect-badge effect-${data.type} ${animClass}">
                <div class="effect-header">
                    <i class="fa-solid ${data.icon}"></i>
                    <span>${effectName}</span>
                </div>
                <div class="effect-desc">
                    ${data.desc}
                </div>
            </div>
        `;
    }
    
    // 4. Montagem da Cena
    const scene = document.createElement('div');
    scene.className = 'combat-scene';    
    
    scene.innerHTML = `
        ${effectHtml} 

        <div class="combatant ${heroHurt}">
            <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${agent.name}" class="avatar-hero">
            <div style="color:#a29bfe; margin-top:5px; font-weight:bold; font-size:12px;">${agent.name}</div>
        </div>
        
        <div class="vs-text">VS</div>
        
        <div class="combatant ${enemyHurt}">
            <img src="https://api.dicebear.com/7.x/bottts/svg?seed=${enemyName}" class="avatar-enemy" style="filter: hue-rotate(320deg);">
            
            <div class="enemy-stats">
                <div class="enemy-name">${enemyName}</div>
                <div class="enemy-bar-bg">
                    <div class="enemy-bar-fill" style="width: ${enemyHpPct}%;"></div>
                </div>
                <span class="enemy-hp-text">${enemyHpText}</span>
            </div>
        </div>
    `;
    container.appendChild(scene);
}

// --- UTILITÁRIOS ---
function createCard(text, icons, type) {
    const card = document.createElement('div');
    card.className = `room-card ${type === 'hidden' ? 'card-hidden' : ''}`;
    
    const iconCont = document.createElement('div');
    iconCont.className = 'card-icons-container';
    
    if(type === 'hidden') {
        iconCont.innerHTML = '<i class="fa-solid fa-question icon-unknown" style="font-size:20px"></i>';
    } else {
        icons.slice(0, 3).forEach(i => {
            const el = document.createElement('i');
            el.className = `fa-solid ${i.class} mini-icon ${i.color}`;
            if(i.tip) el.title = i.tip;
            iconCont.appendChild(el);
        });
    }
    
    card.appendChild(iconCont);
    
    const lbl = document.createElement('span');
    lbl.className = 'room-label';
    lbl.innerText = text;
    card.appendChild(lbl);
    
    return card;
}

// Formata IDs de sala para nomes legíveis
function formatRoomName(rawId) {
    if (!rawId) return "UNKNOWN";
    // Ex: p_1_1 -> ZONE 1-1
    if (rawId.startsWith("p_")) {
        const parts = rawId.split('_');
        return `ZONE ${parts[1]}-${parts[2]}`;
    }
    // Ex: k_5_5 -> ARENA 5-5
    if (rawId.startsWith("k_")) {
        const parts = rawId.split('_');
        return `ARENA ${parts[1]}-${parts[2]}`;
    }
    if (rawId.toLowerCase() === 'start') return "ENTRY POINT";
    return rawId.toUpperCase();
}

function detectEnemyRank(name) {
    if(!name) return 'blue';
    const n = name.toLowerCase();
    if(n.includes('king') || n.includes('lord') || n.includes('boss')) return 'red';
    if(n.includes('elite') || n.includes('captain') || n.includes('(x)')) return 'yellow';
    return 'blue';
}


// 3. CONTROLES DE REPRODUÇÃO
// Configurações de Velocidade
const SPEEDS = [1, 1.5, 2, 4];
let speedIndex = 0;
let baseDelay = 1000; // 1 segundo por turno (na velocidade 1x)

function getDelay() {
    return baseDelay / SPEEDS[speedIndex];
}

function toggleSpeed() {
    // Alterna o índice (0 -> 1 -> 2 -> 3 -> 0)
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    const currentSpeed = SPEEDS[speedIndex];
    
    // Atualiza texto do botão
    const btn = document.getElementById('btn-speed');
    btn.innerText = `${currentSpeed}x`;
    
    // Se estiver tocando, reinicia o timer com a nova velocidade
    if (isPlaying) {
        clearInterval(playInterval);
        playInterval = setInterval(() => step(1), getDelay());
    }
}

function play() {
    if (isPlaying) return;
    isPlaying = true;
    
    // Atualiza ícones (opcional, visual)
    document.getElementById('btn-play').innerHTML = '<i class="fa-solid fa-play" style="color:#55efc4"></i>';
    
    // Inicia o loop com a velocidade atual
    playInterval = setInterval(() => step(1), getDelay());
}

function pause() {
    isPlaying = false;
    clearInterval(playInterval);
    
    // Reseta ícone
    document.getElementById('btn-play').innerHTML = '<i class="fa-solid fa-play"></i>';
}

function step(dir) {
    const next = currentFrame + dir;
    if (next >= 0 && next < replayData.length) {
        currentFrame = next;
        renderFrame(currentFrame);
    } else {
        pause(); // Chegou no fim ou início
    }
}

// --- EVENT LISTENERS ---
document.getElementById('btn-play').onclick = play;
document.getElementById('btn-pause').onclick = pause;
document.getElementById('btn-next').onclick = () => { pause(); step(1); };
document.getElementById('btn-prev').onclick = () => { pause(); step(-1); };
document.getElementById('btn-speed').onclick = toggleSpeed; // <--- Novo Listener

// Listener da Timeline (Slider)
const timeline = document.getElementById('timeline');
timeline.oninput = (e) => {
    pause(); // Pausa ao arrastar manualmente
    currentFrame = parseInt(e.target.value);
    renderFrame(currentFrame);
};

// 4. BOOT E CORREÇÃO DA TIMELINE
function initSystem(data) {
    replayData = data;
    console.log(`Dados carregados: ${replayData.length} frames.`);

    // [CORREÇÃO] Configura o slider IMEDIATAMENTE após carregar os dados
    const timeline = document.getElementById('timeline');
    timeline.min = 0;
    timeline.max = replayData.length - 1; // Garante que vá até o último frame
    timeline.value = 0;

    // Renderiza o primeiro frame
    renderFrame(0);
}

// Fetch Inicial
fetch(RECORD_PATH)
    .then(r => r.ok ? r.json() : null)
    .then(d => {
        if (d) {
            console.log("Modo: Arquivo Real");
            initSystem(d);
        } else {
            console.warn("Modo: Mock Data (Arquivo não encontrado)");
            initSystem(mockReplay);
        }
    })
    .catch(e => {
        console.error("Erro fatal:", e);
        initSystem(mockReplay);
    });