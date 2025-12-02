// 1. DADOS DE EXEMPLO (Substitua pelo fetch do seu JSON)

const mockReplay = [
    {
        turn: 1,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0, imag: 0},
                location_node: "p_1_0", floor: 1, scene_mode: "EXPLORATION",
                logs: ["Entrou no andar 1."],
                neighbors: [
                    {id: "p_1_1", has_enemy: true, enemy_type: "Goblin", has_treasure: false},
                    {id: "p_1_2", has_enemy: false, has_treasure: true},                    
                ]
            }
        }
    },
    {
        turn: 2,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0.1, imag: 0.1},
                location_node: "p_1_1", floor: 1, scene_mode: "COMBAT_PVE",
                combat_enemy: "Goblin",
                logs: ["Encontrou um Goblin!", "Iniciando Combate."],
                neighbors: []
            }
        }
    },
    {
        turn: 3,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 90, max_hp: 100, level: 1, karma: {real: 0.1, imag: 0.1},
                location_node: "p_1_1", floor: 1, scene_mode: "COMBAT_PVE",
                combat_enemy: "Goblin",
                logs: ["Sofreu 10 de dano do Goblin."],
                neighbors: []
            }
        }
    }
];

// Configuração Global
let replayData = mockReplay; // No futuro: await fetch('replay.json')
let currentFrame = 0;
let isPlaying = false;
let playInterval;
const agentId = "a1"; // Focando no Agente 1 para visualização

// 2. FUNÇÕES DE RENDERIZAÇÃO

function renderFrame(index) {
    if (index >= replayData.length) {
        pause();
        return;
    }
    
    const frame = replayData[index];
    const agent = frame.agents[agentId];
    
    // Atualiza Header
    document.getElementById('floor-display').innerText = `FLOOR ${agent.floor}`;
    document.getElementById('turn-display').innerText = frame.turn;
    
    // Atualiza HUD (Stats)
    updateStats(agent);
    
    // Atualiza Karma
    drawKarma(agent.karma);
    
    // Atualiza Logs
    updateLogs(agent.logs);
    
    // Renderiza a Cena Principal
    const stage = document.getElementById('stage');
    
    if (agent.scene_mode.includes("COMBAT")) {
        renderCombatScene(stage, agent, index);
    } else {
        renderExplorationScene(stage, agent);
    }
    
    // Atualiza Slider
    document.getElementById('timeline').value = index;
}

function updateStats(agent) {
    // HP
    const hpPct = (agent.hp / agent.max_hp) * 100;
    document.getElementById('hp-bar').style.width = `${hpPct}%`;
    document.getElementById('hp-text').innerText = `${Math.ceil(agent.hp)}/${agent.max_hp}`;
    
    // XP (Simulado para demo, pegue do JSON real)
    document.getElementById('xp-bar').style.width = `45%`; 
    document.getElementById('xp-text').innerText = `LVL ${agent.level}`;
}

function drawKarma(karma) {
    const canvas = document.getElementById('karma-disk');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const r = w / 2;
    
    // Limpa
    ctx.clearRect(0, 0, w, h);
    
    // Desenha Disco
    ctx.beginPath();
    ctx.arc(r, r, r - 2, 0, 2 * Math.PI);
    ctx.strokeStyle = '#555';
    ctx.stroke();
    ctx.fillStyle = '#222';
    ctx.fill();
    
    // Desenha Ponto
    // Coordenadas vem de -1 a 1. Mapear para canvas.
    const x = r + (karma.real * (r - 5));
    const y = r + (karma.imag * (r - 5)); // Inverter Y se necessário
    
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = karma.real < -0.5 ? '#ff7675' : (karma.real > 0.5 ? '#55efc4' : '#ffeaa7');
    ctx.fill();
}

function updateLogs(logs) {
    const container = document.getElementById('log-container');
    container.innerHTML = '';
    logs.forEach(log => {
        const div = document.createElement('div');
        div.className = 'log-entry';
        div.innerText = `> ${log}`;
        container.appendChild(div);
    });
}

// --- CENA DE EXPLORAÇÃO ---
function renderExplorationScene(container, agent) {
    container.innerHTML = ''; 
    
    const colContainer = document.createElement('div');
    colContainer.className = 'exploration-container';
    
    // --- TIER 1: FUTURO (Oculto) ---
    const hiddenRow = document.createElement('div');
    hiddenRow.className = 'row-tier';
    for(let i=0; i<4; i++) {
        // Card oculto sem ícones específicos
        hiddenRow.appendChild(createCard('Oculto', [], 'hidden'));
    }
    colContainer.appendChild(hiddenRow);

    // --- TIER 2: OPÇÕES (Vizinhos) ---
    const choicesRow = document.createElement('div');
    choicesRow.className = 'row-tier';
    
    if (agent.neighbors && agent.neighbors.length > 0) {
        agent.neighbors.forEach(n => {
            // Lista de ícones para esta sala
            let iconsList = [];
            
            // 1. INIMIGO
            if (n.has_enemy) {
                const rank = detectEnemyRank(n.enemy_type);
                iconsList.push({
                    class: 'fa-skull', 
                    colorClass: `skull-${rank}`, // skull-blue, skull-red, etc.
                    tooltip: `Inimigo: ${n.enemy_type} (${rank})`
                });
            }
            
            // 2. TESOURO / EVENTO
            if (n.has_treasure) {
                iconsList.push({
                    class: 'fa-gem', // ou fa-scroll para evento
                    colorClass: 'icon-event',
                    tooltip: 'Tesouro ou Evento'
                });
            }
            
            // 3. EFEITO DE SALA
            if (n.effect && n.effect !== 'None') {
                iconsList.push({
                    class: 'fa-wind', // Ícone genérico de efeito
                    colorClass: 'icon-effect',
                    tooltip: `Efeito: ${n.effect}`
                });
            }
            
            // Se estiver vazio
            if (iconsList.length === 0 && !n.is_exit) {
                iconsList.push({ class: 'fa-road', colorClass: 'icon-unknown', tooltip: 'Vazio' });
            }
            // Se for saída
            if (n.is_exit) {
                 iconsList.push({ class: 'fa-door-open', colorClass: 'icon-door', tooltip: 'Saída' });
            }

            // Cria o card passando a lista de ícones
            choicesRow.appendChild(createCard(n.id || '?', iconsList, 'choice'));
        });
    } else {
        choicesRow.appendChild(createCard('Parede', [{class: 'fa-ban', colorClass: 'icon-unknown'}], 'wall'));
    }
    colContainer.appendChild(choicesRow);

    // --- TIER 3: AGENTE ---
    const agentRow = document.createElement('div');
    agentRow.className = 'row-tier';
    // Agente tem seu próprio ícone fixo
    const agentIcons = [{class: 'fa-user-ninja', colorClass: 'icon-agent', tooltip: 'Você'}];
    const agentCard = createCard(agent.name || 'Heroi', agentIcons, 'self');
    agentCard.classList.add('card-agent');
    agentRow.appendChild(agentCard);
    
    colContainer.appendChild(agentRow);
    container.appendChild(colContainer);
}

// --- FUNÇÃO AUXILIAR PARA CRIAR O CARD COM MÚLTIPLOS ÍCONES ---
function createCard(title, iconsList, type) {
    const card = document.createElement('div');
    card.className = 'room-card';
    if (type === 'hidden') card.classList.add('card-hidden');

    // Cria o container flex para os ícones
    const iconsContainer = document.createElement('div');
    iconsContainer.className = 'card-icons-container';

    // Se for oculto, põe só uma interrogação
    if (type === 'hidden') {
        iconsContainer.innerHTML = '<i class="fa-solid fa-question mini-icon icon-unknown"></i>';
    } else {
        // Renderiza cada ícone da lista (até 3)
        iconsList.slice(0, 3).forEach(icon => {
            const iTag = document.createElement('i');
            iTag.className = `fa-solid ${icon.class} mini-icon ${icon.colorClass}`;
            if(icon.tooltip) iTag.title = icon.tooltip;
            iconsContainer.appendChild(iTag);
        });
    }

    // Monta o HTML final
    card.appendChild(iconsContainer);
    
    // Label da sala (ID)
    const label = document.createElement('span');
    label.className = 'room-label';
    label.innerText = title;
    card.appendChild(label);

    return card;
}

// --- LÓGICA DE RANK DO INIMIGO ---
function detectEnemyRank(name) {
    if (!name) return 'blue'; // Padrão
    const n = name.toLowerCase();
    
    // Defina aqui suas palavras-chave para Chefes
    if (n.includes('king') || n.includes('lord') || n.includes('boss') || n.includes('dragon')) {
        return 'red';
    }
    
    // Defina aqui suas palavras-chave para Elite
    if (n.includes('elite') || n.includes('captain') || n.includes('(x)') || n.includes('knight')) {
        return 'yellow';
    }
    
    return 'blue'; // Comum
}

// --- CENA DE COMBATE ---
function renderCombatScene(container, agent, index) {
    container.innerHTML = '';
    
    const scene = document.createElement('div');
    scene.className = 'combat-scene';
    
    // Verifica se houve dano (para animar shake)
    const prevFrame = replayData[index - 1];
    const isHurt = prevFrame && prevFrame.agents[agentId].hp > agent.hp;
    const shakeClass = isHurt ? 'shake' : '';
    
    scene.innerHTML = `
        <div class="combatant ${shakeClass}">
            <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=Felix" alt="Hero">
            <div style="margin-top: 5px; color: #a29bfe;">HERO</div>
        </div>
        
        <div class="vs-text">VS</div>
        
        <div class="combatant">
            <img src="https://api.dicebear.com/7.x/bottts/svg?seed=${agent.combat_enemy}" alt="Enemy" style="filter: hue-rotate(90deg);">
            <div style="margin-top: 5px; color: #ff7675;">${agent.combat_enemy}</div>
        </div>
    `;
    
    container.appendChild(scene);
}

// 3. CONTROLE DE PLAYBACK

function play() {
    if (isPlaying) return;
    isPlaying = true;
    playInterval = setInterval(() => {
        currentFrame++;
        renderFrame(currentFrame);
    }, 1000); // 1 segundo por turno
}

function pause() {
    isPlaying = false;
    clearInterval(playInterval);
}

// Event Listeners
document.getElementById('btn-play').addEventListener('click', play);
document.getElementById('btn-pause').addEventListener('click', pause);
document.getElementById('timeline').addEventListener('input', (e) => {
    pause();
    currentFrame = parseInt(e.target.value);
    renderFrame(currentFrame);
});

function stepForward() {
    pause(); // Para a reprodução automática
    if(currentFrame < replayData.length - 1) {
        currentFrame++;
        renderFrame(currentFrame);
        document.getElementById('timeline').value = currentFrame;
    }
}

function stepBackward() {
    pause();
    if(currentFrame > 0) {
        currentFrame--;
        renderFrame(currentFrame);
        document.getElementById('timeline').value = currentFrame;
    }
}

// Event Listeners para os novos botões
document.getElementById('btn-next').addEventListener('click', stepForward);
document.getElementById('btn-prev').addEventListener('click', stepBackward);

// 4. INICIALIZAÇÃO INTELIGENTE (FETCH OU MOCK)

const RECORD_PATH = 'records/replay.json'; // Caminho da pasta solicitado

function startVisualizer() {
    // Configura o slider com base no tamanho dos dados carregados
    const timeline = document.getElementById('timeline');
    timeline.max = replayData.length - 1;
    timeline.value = 0;
    
    // Renderiza o primeiro frame
    renderFrame(0);
    
    console.log(`Visualizador iniciado com ${replayData.length} frames.`);
}

// Tenta carregar o arquivo real
fetch(RECORD_PATH)
    .then(response => {
        if (!response.ok) {
            throw new Error("Arquivo de replay não encontrado na pasta records/");
        }
        return response.json();
    })
    .then(data => {
        console.log("✅ Sucesso: Replay carregado de " + RECORD_PATH);
        replayData = data; // Sobrescreve o mock com dados reais
        startVisualizer();
    })
    .catch(error => {
        console.warn("⚠️ Aviso: Não foi possível carregar " + RECORD_PATH);
        console.warn("ℹ️ Motivo: " + error.message);
        console.log("🔄 Usando MOCK DATA para teste.");
        
        // Mantém o mockReplay que já estava definido no início do arquivo
        replayData = mockReplay; 
        startVisualizer();
    });

// Inicialização (Simula carregamento do arquivo)
document.getElementById('timeline').max = mockReplay.length - 1;
renderFrame(0);

