// 1. INICIALIZAÇÃO E DADOS

const RECORD_PATH = 'records/replay.json';
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

    const stage = document.getElementById('stage');
    if (agent.scene_mode.includes("COMBAT")) {
        renderCombatScene(stage, agent, index);
    } else {
        renderExplorationScene(stage, agent);
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
    // Passa o objeto de cooldowns do JSON. Se não existir, passa vazio.
    updateSkills(agent.cooldowns || {});
}

// --- FUNÇÃO DE ATUALIZAÇÃO DE SKILLS ---
function updateSkills(cooldowns) {
    // Mapeamento fixo das skills para os IDs do HTML
    // A ordem importa: 0=Quick, 1=Heavy, 2=Shield, 3=Wait
    const skillMap = [
        { name: "Quick Strike", id: "skill-0" },
        { name: "Heavy Blow",   id: "skill-1" },
        { name: "Stone Shield", id: "skill-2" },
        { name: "Wait",         id: "skill-3" }
    ];

    skillMap.forEach(skill => {
        const element = document.getElementById(skill.id);
        const overlay = element.querySelector('.cd-overlay');
        
        // Pega o valor do CD atual (default 0 se não vier no JSON)
        const currentCD = cooldowns[skill.name] || 0;

        if (currentCD > 0) {
            // ESTÁ EM COOLDOWN
            element.classList.add('on-cooldown');
            overlay.innerText = currentCD; // Mostra o número branco no centro
        } else {
            // ESTÁ DISPONÍVEL
            element.classList.remove('on-cooldown');
            overlay.innerText = "";
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
function renderExplorationScene(container, agent) {
    container.innerHTML = '';
    const col = document.createElement('div');
    col.className = 'exploration-container';

    // Tier 1: Oculto
    const row1 = document.createElement('div');
    row1.className = 'row-tier';
    for(let i=0; i<4; i++) row1.appendChild(createCard('Oculto', [], 'hidden'));
    col.appendChild(row1);

    // Tier 2: Vizinhos
    const row2 = document.createElement('div');
    row2.className = 'row-tier';
    
    if (agent.neighbors && agent.neighbors.length > 0) {
        agent.neighbors.forEach(n => {
            let icons = [];
            
            // Inimigo (Cor por Rank)
            if (n.has_enemy) {
                const rank = detectEnemyRank(n.enemy_type);
                icons.push({
                    class: 'fa-skull', 
                    color: `skull-${rank}`, 
                    tip: `Inimigo: ${n.enemy_type}`
                });
            }
            // Tesouro
            if (n.has_treasure) {
                icons.push({class: 'fa-gem', color: 'icon-event', tip: 'Tesouro/Evento'});
            }
            // Efeito (Armadilha invisível para o agente, visível aqui)
            if (n.effect && n.effect !== 'None') {
                icons.push({class: 'fa-wind', color: 'icon-effect', tip: `Efeito: ${n.effect}`});
            }
            // Saída
            if (n.is_exit) {
                icons.push({class: 'fa-door-open', color: 'icon-door', tip: 'Saída'});
            }
            // Vazio
            if (icons.length === 0 && !n.is_exit) {
                icons.push({class: 'fa-road', color: 'icon-unknown', tip: 'Vazio'});
            }

            row2.appendChild(createCard(n.id || '?', icons, 'choice'));
        });
    } else {
        row2.appendChild(createCard('Parede', [{class: 'fa-ban', color: 'icon-unknown'}], 'wall'));
    }
    col.appendChild(row2);

    // Tier 3: Agente
    const row3 = document.createElement('div');
    row3.className = 'row-tier';
    const agentIcons = [{class: 'fa-user-ninja', color: 'icon-agent', tip: 'Você'}];
    const agCard = createCard(agent.name, agentIcons, 'self');
    agCard.classList.add('card-agent');
    row3.appendChild(agCard);
    
    col.appendChild(row3);
    container.appendChild(col);
}

// --- NOVA FUNÇÃO DE EQUIPAMENTO ---
function updateEquipmentSlots(equipmentList) {
    // Assume que a lista vem na ordem [Arma, Armadura, Artefato] ou busca por keywords    
    
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
    el.className = 'equip-slot border-common'; // Volta pro padrão (verde ou cinza se preferir)
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
    const prev = replayData[index-1];
    // Shake se HP caiu
    const hurt = prev && prev.agents[Object.keys(prev.agents)[0]].hp > agent.hp ? 'shake' : '';
    
    const scene = document.createElement('div');
    scene.className = 'combat-scene';
    scene.innerHTML = `
        <div class="combatant ${hurt}">
            <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${agent.name}" class="avatar-hero">
            <div style="color:#a29bfe; margin-top:10px;">${agent.name}</div>
        </div>
        <div class="vs-text">VS</div>
        <div class="combatant">
            <img src="https://api.dicebear.com/7.x/bottts/svg?seed=${agent.combat_enemy}" class="avatar-enemy" style="filter: hue-rotate(45deg);">
            <div style="color:#ff7675; margin-top:10px;">${agent.combat_enemy}</div>
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

function detectEnemyRank(name) {
    if(!name) return 'blue';
    const n = name.toLowerCase();
    if(n.includes('king') || n.includes('lord') || n.includes('boss')) return 'red';
    if(n.includes('elite') || n.includes('captain') || n.includes('(x)')) return 'yellow';
    return 'blue';
}

// 3. CONTROLES
function play() { if(!isPlaying) { isPlaying=true; playInterval=setInterval(()=>step(1), 800); } }
function pause() { isPlaying=false; clearInterval(playInterval); }
function step(dir) {
    const next = currentFrame + dir;
    if(next >= 0 && next < replayData.length) {
        currentFrame = next;
        renderFrame(currentFrame);
    } else {
        pause();
    }
}

document.getElementById('btn-play').onclick = play;
document.getElementById('btn-pause').onclick = pause;
document.getElementById('btn-next').onclick = () => { pause(); step(1); };
document.getElementById('btn-prev').onclick = () => { pause(); step(-1); };
document.getElementById('timeline').oninput = (e) => { pause(); currentFrame = +e.target.value; renderFrame(currentFrame); };

// 4. BOOT
fetch(RECORD_PATH).then(r=>r.ok?r.json():null).then(d => {
    if(d) { replayData = d; console.log("Loaded Real Data"); }
    else { replayData = mockReplay; console.log("Loaded Mock Data"); }
    document.getElementById('timeline').max = replayData.length - 1;
    renderFrame(0);
}).catch(e => {
    console.log("Error loading, using mock.", e);
    replayData = mockReplay;
    renderFrame(0);
});