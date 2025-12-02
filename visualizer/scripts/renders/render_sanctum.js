// scripts/render_sanctum.js
function renderSanctuaryScene(container, frame) {
    container.innerHTML = '';

    // 1. Título
    const titleDiv = document.createElement('div');
    titleDiv.className = 'room-title-display';
    titleDiv.innerHTML = '<span style="color:var(--accent-blood)">SANCTUM ZONE</span>';
    container.appendChild(titleDiv);

    // 2. Wrapper
    const sanctumWrapper = document.createElement('div');
    sanctumWrapper.className = 'sanctuary-container';

    // 3. Camadas
    const svgLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgLayer.classList.add("sanctuary-svg");
    sanctumWrapper.appendChild(svgLayer);

    const gridLayer = document.createElement('div');
    gridLayer.className = 'sanctuary-grid';
    
    // 4. Lógica de Ocupação e Arestas
    const nodeOccupancy = {}; 
    const itemsOnFloor = {}; 
    let edgesToDraw = [];

    // A. Identifica quem estamos assistindo (Câmera)
    const cameraAgentId = window.selectedAgentId || Object.keys(frame.agents)[0];
    const cameraAgent = frame.agents[cameraAgentId];

    // B. Define quem deve aparecer (Eu + Meu Oponente)
    const visibleAgentsIds = [cameraAgentId];
    if (cameraAgent && cameraAgent.opponent_id) {
        visibleAgentsIds.push(cameraAgent.opponent_id);
    }

    // C. Pega o mapa da Arena (usando a visão da câmera)
    if (cameraAgent && cameraAgent.arena_edges) {
        edgesToDraw = cameraAgent.arena_edges;
    }

    // D. Processa apenas os agentes visíveis
    visibleAgentsIds.forEach(agentId => {
        const ag = frame.agents[agentId];
        
        // Segurança: se o oponente não existir no frame ou não estiver na arena
        if (!ag) return; 
        
        // Verifica se está visualmente na arena (nó k_...)
        const isInSanctumNode = ag.location_node && ag.location_node.startsWith('k_');
        if (!isInSanctumNode && ag.scene_mode !== 'ARENA' && ag.scene_mode !== 'COMBAT_PVP') return;

        const nodeIdx = parseKNodeIndex(ag.location_node);
        
        if (nodeIdx !== null) {
            // Adiciona Agente ao Nó
            if (!nodeOccupancy[nodeIdx]) nodeOccupancy[nodeIdx] = [];
            nodeOccupancy[nodeIdx].push({...ag, id: agentId});

            // Adiciona Itens do Chão (apenas se for a câmera ou relevante)
            if (ag.room_items && ag.room_items.length > 0) {
                itemsOnFloor[nodeIdx] = ag.room_items;
            }
        }
    });
    // 5. Desenha Linhas (Baseado no JSON real do mapa)
    edgesToDraw.forEach(edge => {
        const [u, v] = edge;
        const line = createSvgLine(u, v);
        svgLayer.appendChild(line);
    });

    // 6. Desenha Nós (0 a 8)
    for (let i = 0; i < 9; i++) {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = `k-node`;
        const floorDiv = document.createElement('div');
        floorDiv.className = 'k-room-floor';
        
        // Verifica saída 
        // Se houve encontro, a sala atual brilha
        const hasAgents = nodeOccupancy[i] && nodeOccupancy[i].length > 0;
        const isMeeting = nodeOccupancy[i] && nodeOccupancy[i].length > 1;

        if (hasAgents) {
            nodeDiv.classList.add('has-agent');
            if (isMeeting) nodeDiv.classList.add('has-exit'); // Encontro = Brilho

            // Desenha Agentes
            nodeOccupancy[i].forEach((ag, idx) => {
                const img = document.createElement('img');
                if (window.AssetManager) {
                    img.src = AssetManager.getAgentSprite(ag.name);
                }
                img.className = 'k-avatar';
                
                // Offset para não sobrepor se houver 2
                if (nodeOccupancy[i].length > 1) {
                    const offset = idx === 0 ? -10 : 10;
                    img.style.transform = `translateX(${offset}px)`;
                    img.style.zIndex = idx + 10;
                }
                floorDiv.appendChild(img);
            });
        }

        // Desenha Itens Dropados (Ação Social)
        if (itemsOnFloor[i]) {
            const itemIcon = document.createElement('div');
            itemIcon.className = 'k-item';
            // Se for artefato, ícone de anel, senão baú
            const isArtifact = itemsOnFloor[i].some(it => it.includes('Amulet') || it.includes('Ring'));
            itemIcon.innerHTML = `<i class="fa-solid ${isArtifact ? 'fa-ring' : 'fa-box-open'}"></i>`;
            itemIcon.title = `Chão: ${itemsOnFloor[i].join(', ')}`;
            floorDiv.appendChild(itemIcon);
        }

        nodeDiv.appendChild(floorDiv);
        gridLayer.appendChild(nodeDiv);
    }

    sanctumWrapper.appendChild(gridLayer);
    container.appendChild(sanctumWrapper);
}

// Auxiliares
function createSvgLine(u, v) {
    const getCoord = (idx) => {
        const row = Math.floor(idx / 3);
        const col = idx % 3;
        // Centros: 16.66%, 50%, 83.33%
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
    if (!nodeId || !nodeId.startsWith('k_')) return null;
    return parseInt(nodeId.split('_')[2]);
}