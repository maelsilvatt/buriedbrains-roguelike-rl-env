// render_exploration.js

function renderExplorationScene(container, agent, index) {
    container.innerHTML = '';
    
    const titleDiv = document.createElement('div');
    titleDiv.className = 'room-title-display';
    titleDiv.innerText = `CURRENT: ${formatRoomName(agent.location_node)}`;
    container.appendChild(titleDiv);

    const col = document.createElement('div');
    col.className = 'exploration-container';

    // Tier 1: Oculto
    const row1 = document.createElement('div');
    row1.className = 'row-tier';
    for(let i=0; i<4; i++) row1.appendChild(createCard('UNKNOWN', [], 'hidden'));
    col.appendChild(row1);

    // Tier 2: Vizinhos
    let chosenNodeId = null;
    // Tenta acessar replayData globalmente (vinda do main.js)
    if (window.replayData && index < window.replayData.length - 1) {
        const nextFrame = window.replayData[index + 1];
        const nextAgentId = Object.keys(nextFrame.agents).find(k => nextFrame.agents[k].name === agent.name) || Object.keys(nextFrame.agents)[0];
        if (nextAgentId) chosenNodeId = nextFrame.agents[nextAgentId].location_node;
    }

    const row2 = document.createElement('div');
    row2.className = 'row-tier';
    
    if (agent.neighbors && agent.neighbors.length > 0) {
        agent.neighbors.forEach(n => {
            let icons = [];
            if (n.has_enemy) {
                const rank = detectEnemyRank(n.enemy_type);
                icons.push({ class: 'fa-skull', color: `skull-${rank}`, tip: `Inimigo: ${n.enemy_type}` });
            }
            if (n.has_treasure) icons.push({class: 'fa-gem', color: 'icon-event', tip: 'Tesouro'});
            if (n.effect && n.effect !== 'None') icons.push({class: 'fa-wind', color: 'icon-effect', tip: n.effect});
            if (n.is_exit) icons.push({class: 'fa-door-open', color: 'icon-door', tip: 'Saída'});
            if (icons.length === 0 && !n.is_exit) icons.push({class: 'fa-road', color: 'icon-unknown', tip: 'Vazio'});

            const card = createCard(formatRoomName(n.id), icons, 'choice');
            if (n.id === chosenNodeId) {
                card.classList.add('selected');
                const indicator = document.createElement('div');
                indicator.className = 'selection-indicator';
                indicator.innerText = "CHOSEN";
                card.appendChild(indicator);
            }
            row2.appendChild(card);
        });
    } else {
        row2.appendChild(createCard('WALL', [{class: 'fa-ban', color: 'icon-unknown'}], 'wall'));
    }
    col.appendChild(row2);

    // Tier 3: Agente
    const row3 = document.createElement('div');
    row3.className = 'row-tier';
    const agentIcons = [{class: 'fa-user-ninja', color: 'icon-agent', tip: 'Você'}];
    const agCard = createCard('HERO', agentIcons, 'self');
    agCard.classList.add('card-agent');
    row3.appendChild(agCard);
    
    col.appendChild(row3);
    container.appendChild(col);
}

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