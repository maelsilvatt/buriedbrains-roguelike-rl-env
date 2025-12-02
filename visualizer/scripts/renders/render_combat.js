// render_combat.js

function renderCombatScene(container, agent, index) {
    container.innerHTML = '';
    
    // Dados para cálculo de dano
    const prevFrame = index > 0 && window.replayData ? window.replayData[index - 1] : null;
    let prevAgent = null;
    if (prevFrame) {
        // Tenta achar o mesmo agente no frame anterior
        const prevAgentId = Object.keys(prevFrame.agents).find(k => prevFrame.agents[k].name === agent.name) || Object.keys(prevFrame.agents)[0];
        prevAgent = prevFrame.agents[prevAgentId];
    }
    const heroHurt = prevAgent && prevAgent.hp > agent.hp ? 'shake' : '';
    
    let enemyHurt = '';
    let enemyHpPct = 100;
    let enemyName = "Inimigo";
    let enemyHpText = "??/??";

    if (agent.combat_data) {
        enemyName = agent.combat_data.name;
        const curHp = agent.combat_data.hp;
        const maxHp = agent.combat_data.max_hp;
        enemyHpPct = Math.max(0, (curHp / maxHp) * 100);
        enemyHpText = `${Math.ceil(curHp)}/${Math.ceil(maxHp)}`;

        if (prevAgent && prevAgent.combat_data && prevAgent.combat_data.hp > curHp) {
            enemyHurt = 'shake';
        }
    }
    
    const heroImgSrc = window.AssetManager ? AssetManager.getAgentSprite(agent.name) : '';
    const enemyImgSrc = window.AssetManager ? AssetManager.getEnemyImage(enemyName) : '';

    // Efeito
    let effectHtml = '';
    const effectName = agent.current_effect;
    let wasInCombatBefore = false;
    if (prevAgent && prevAgent.scene_mode && prevAgent.scene_mode.includes("COMBAT")) {
        wasInCombatBefore = true;
    }

    if (effectName && effectName !== 'None' && window.EFFECT_LIBRARY && EFFECT_LIBRARY[effectName]) {
        const data = EFFECT_LIBRARY[effectName];
        const animClass = wasInCombatBefore ? '' : 'anim-enter';
        effectHtml = `
            <div class="combat-effect-badge effect-${data.type} ${animClass}">
                <div class="effect-header"><i class="fa-solid ${data.icon}"></i><span>${effectName}</span></div>
                <div class="effect-desc">${data.desc}</div>
            </div>
        `;
    }
    
    const scene = document.createElement('div');
    scene.className = 'combat-scene';
    scene.innerHTML = `
        ${effectHtml} 
        <div class="combatant ${heroHurt}">
            <img src="${heroImgSrc}" class="avatar-hero">
            <div style="color:var(--accent-soul); margin-top:5px; font-weight:bold; font-size:12px;">${agent.name}</div>
        </div>
        <div class="vs-text">VS</div>
        <div class="combatant ${enemyHurt}">
            <img src="${enemyImgSrc}" class="avatar-enemy">
            <div class="enemy-stats">
                <div class="enemy-name">${enemyName}</div>
                <div class="enemy-bar-bg"><div class="enemy-bar-fill" style="width: ${enemyHpPct}%;"></div></div>
                <span class="enemy-hp-text">${enemyHpText}</span>
            </div>
        </div>
    `;
    container.appendChild(scene);
}