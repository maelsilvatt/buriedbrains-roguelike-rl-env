// ui_hud.js - Interface do Usuário (Rodapé e Logs)

function updateStats(agent, actionName) {
    // Barras
    const hpPct = Math.max(0, (agent.hp / agent.max_hp) * 100);
    const xpPct = agent.exp_percent || 0;
    
    document.getElementById('hp-bar').style.width = `${hpPct}%`;
    document.getElementById('hp-text').innerText = `${Math.ceil(agent.hp)}/${agent.max_hp}`;
    document.getElementById('xp-bar').style.width = `${xpPct}%`;
    document.getElementById('xp-text').innerText = `XP ${xpPct}%`;
    
    // Avatar
    const avatar = document.getElementById('agent-avatar');
    if (window.AssetManager) {
        avatar.src = AssetManager.getAgentPortrait(agent.name);
    }
    if(agent.hp <= 0) avatar.style.filter = "grayscale(100%)";
    else avatar.style.filter = "none";

    // Sub-componentes
    updateEquipmentSlots(agent.equipment);
    if (agent.cooldowns) {
        updateSkills(agent.cooldowns, actionName);
    }
}

function updateEquipmentSlots(equipmentList) {
    const slots = {
        weapon: document.getElementById('slot-weapon'),
        armor: document.getElementById('slot-armor'),
        artifact: document.getElementById('slot-artifact')
    };

    resetSlot(slots.weapon, 'fa-fist-raised');
    resetSlot(slots.armor, 'fa-shirt');
    resetSlot(slots.artifact, 'fa-ring');

    if (!equipmentList) return;

    equipmentList.forEach(item => {
        if(!item) return;
        const name = item.toLowerCase();
        let targetSlot = null;
        let iconClass = '';

        if (name.includes('sword') || name.includes('bow') || name.includes('axe') || name.includes('staff') || name.includes('dagger') || name.includes('spear')) {
            targetSlot = slots.weapon; iconClass = 'fa-khanda';
        } else if (name.includes('armor') || name.includes('plate') || name.includes('robe') || name.includes('chain') || name.includes('tunic')) {
            targetSlot = slots.armor; iconClass = 'fa-shield-halved';
        } else if (name.includes('amulet') || name.includes('ring') || name.includes('charm') || name.includes('idol')) {
            targetSlot = slots.artifact; iconClass = 'fa-gem';
        }

        if (targetSlot) {
            const rarity = detectRarity(name);
            targetSlot.className = `equip-slot equipped border-${rarity}`;
            targetSlot.title = item;
            targetSlot.innerHTML = `<i class="fa-solid ${iconClass}"></i>`;
        }
    });
}

function resetSlot(el, defaultIcon) {
    el.className = 'equip-slot border-common';
    el.innerHTML = `<i class="fa-solid ${defaultIcon}"></i>`;
    el.title = "Vazio";
    el.classList.remove('equipped');
}

function detectRarity(itemName) {
    const n = itemName.toLowerCase();
    if (n.includes('legendary') || n.includes('godly') || n.includes('ancient') || n.includes('dragon')) return 'legendary';
    if (n.includes('epic') || n.includes('obsidian') || n.includes('crystal') || n.includes('shadow')) return 'epic';
    if (n.includes('rare') || n.includes('steel') || n.includes('silver') || n.includes('reinforced')) return 'rare';
    return 'common';
}

const SKILL_INFO = {
    "Quick Strike": { id: "skill-0", maxCD: 1 },
    "Heavy Blow":   { id: "skill-1", maxCD: 3 },
    "Stone Shield": { id: "skill-2", maxCD: 3 },
    "Wait":         { id: "skill-3", maxCD: 0 }
};

function updateSkills(cooldowns, currentAction) {
    Object.keys(SKILL_INFO).forEach(skillName => {
        const info = SKILL_INFO[skillName];
        const element = document.getElementById(info.id);
        if(!element) return;

        const overlay = element.querySelector('.cd-overlay');
        let cdValue = cooldowns[skillName] || 0;

        if (currentAction === skillName && info.maxCD > 0) {
            cdValue = info.maxCD;
        }

        if (cdValue > 0) {
            element.classList.add('on-cooldown');
            if(overlay) { overlay.style.opacity = '1'; overlay.innerText = cdValue; }
        } else {
            element.classList.remove('on-cooldown');
            if(overlay) { overlay.style.opacity = '0'; overlay.innerText = ""; }
        }
    });
}

function drawKarma(karma) {
    const canvas = document.getElementById('karma-disk');
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, r = w/2;
    
    ctx.clearRect(0,0,w,h);
    ctx.beginPath(); ctx.arc(r, r, r-2, 0, 2*Math.PI);
    ctx.fillStyle = '#111'; ctx.strokeStyle = '#333';
    ctx.fill(); ctx.stroke();
    
    const kx = r + (karma.real * (r-8));
    const ky = r - (karma.imag * (r-8));
    
    ctx.beginPath(); ctx.arc(kx, ky, 4, 0, 2*Math.PI);
    let color = '#ffeaa7';
    if (karma.real > 0.3) color = '#55efc4';
    if (karma.real < -0.3) color = '#ff7675';
    ctx.fillStyle = color; ctx.fill();
}

function updateLogs(logs) {
    const container = document.getElementById('log-container');
    container.innerHTML = logs.map(l => `<div class="log-entry">> ${l}</div>`).join('');
    container.scrollTop = container.scrollHeight;
}

// Controle de câmera
function setupCameraControl(data, onCameraChange) {
    const select = document.getElementById('agent-select');
    if (!select || !data || data.length === 0) return null;

    select.innerHTML = '';
    const firstFrame = data[0];
    
    // Popula opções
    Object.keys(firstFrame.agents).forEach(id => {
        const option = document.createElement('option');
        option.value = id;
        option.innerText = firstFrame.agents[id].name;
        select.appendChild(option);
    });

    // Listener
    select.addEventListener('change', (e) => {
        onCameraChange(e.target.value);
    });

    // Retorna o ID padrão (primeiro da lista)
    return Object.keys(firstFrame.agents)[0];
}