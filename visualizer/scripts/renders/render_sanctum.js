// ==========================================
// CENA DO SANTUÁRIO (GRID 3x3 + SVG)
// ==========================================

function renderSanctuaryScene(container, frame) {
    // Pega o agente focado (definido globalmente no main.js via select)
    // Precisamos acessar a variável global ou passar como argumento.
    // Assumindo que passamos o ID ou o objeto do agente focado:
    const agentSelect = document.getElementById('agent-selector');
    const selectedId = agentSelect ? agentSelect.value : Object.keys(frame.agents)[0];
    const focusedAgent = frame.agents[selectedId];

    if (!focusedAgent) return;

    container.innerHTML = ''; // Limpa palco

    // 1. Título
    const titleDiv = document.createElement('div');
    titleDiv.className = 'room-title-display';
    titleDiv.innerHTML = `<span style="color:#a29bfe">SANCTUM ZONE (Floor ${focusedAgent.floor})</span>`;
    container.appendChild(titleDiv);

    // 2. Estrutura Visual
    const sanctumWrapper = document.createElement('div');
    sanctumWrapper.className = 'sanctuary-container';

    const svgLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgLayer.classList.add("sanctuary-svg");
    sanctumWrapper.appendChild(svgLayer);

    const gridLayer = document.createElement('div');
    gridLayer.className = 'sanctuary-grid';

    // --- LÓGICA DE MAPA (A Correção) ---
    
    // Tenta pegar a configuração da arena do agente focado
    // Se ele estiver na Arena, ele TEM a lista de arestas.
    // Se ele não estiver, procuramos ALGUÉM que esteja na mesma sala (improvável no seu visualizador focado).
    let activeEdges = [];
    let isMeetingActive = false;

    if (focusedAgent.arena_config) {
        activeEdges = focusedAgent.arena_config.edges || [];
        isMeetingActive = focusedAgent.arena_config.meet_occurred || false;
    }

    // A. Desenha as Linhas (Baseado na Verdade do Python)
    activeEdges.forEach(edge => {
        const [u, v] = edge; 
        const line = createSvgLine(u, v);
        svgLayer.appendChild(line);
    });

    // --- LÓGICA DE AGENTES ---
    // Descobre onde todos estão
    const nodeOccupancy = {}; 
    
    Object.values(frame.agents).forEach(ag => {
        // Só desenha agentes que estão no modo ARENA
        if (ag.scene_mode === 'ARENA') {
            const nodeIdx = parseKNodeIndex(ag.location_node);
            if (nodeIdx !== null) {
                if (!nodeOccupancy[nodeIdx]) nodeOccupancy[nodeIdx] = [];
                nodeOccupancy[nodeIdx].push(ag);
            }
        }
    });

    // B. Desenha o Grid 3x3
    for (let i = 0; i < 9; i++) {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = `k-node`;
        
        const floorDiv = document.createElement('div');
        floorDiv.className = 'k-room-floor';
        
        // Se tem gente aqui
        if (nodeOccupancy[i]) {
            nodeDiv.classList.add('has-agent');
            
            // Efeito de Porta Aberta (Encontro)
            if (isMeetingActive && nodeOccupancy[i].length > 1) {
                nodeDiv.classList.add('has-exit'); // Brilho verde
            }

            // Desenha Avatares
            nodeOccupancy[i].forEach((ag, idx) => {
                const img = document.createElement('img');
                img.src = `https://api.dicebear.com/7.x/adventurer/svg?seed=${ag.name}`;
                img.className = 'k-avatar';
                
                // Auras Sociais
                if (ag.social?.offered_peace) img.classList.add('peace-aura');
                if (ag.social?.skipped_attack) img.classList.add('defense-aura');

                // Offset para não encavalar
                if (nodeOccupancy[i].length > 1) {
                    img.style.transform = `translate(${idx * 20 - 10}px, 0)`;
                }
                
                floorDiv.appendChild(img);
            });
        }

        nodeDiv.appendChild(floorDiv);
        gridLayer.appendChild(nodeDiv);
    }

    sanctumWrapper.appendChild(gridLayer);
    container.appendChild(sanctumWrapper);
}

// --- HELPERS ---

function createSvgLine(u, v) {
    const getCoord = (idx) => {
        const row = Math.floor(idx / 3);
        const col = idx % 3;
        // Centraliza no grid 3x3 (16.66% offset)
        return { x: (col * 33.33) + 16.66, y: (row * 33.33) + 16.66 };
    };
    const p1 = getCoord(u);
    const p2 = getCoord(v);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", `${p1.x}%`);
    line.setAttribute("y1", `${p1.y}%`);
    line.setAttribute("x2", `${p2.x}%`);
    line.setAttribute("y2", `${p2.y}%`);
    line.classList.add("connection-line", "active");
    return line;
}

function parseKNodeIndex(nodeId) {
    if (!nodeId || typeof nodeId !== 'string') return null;
    if (!nodeId.startsWith('k_')) return null;
    const parts = nodeId.split('_');
    return parseInt(parts[2]); // k_20_5 -> 5
}