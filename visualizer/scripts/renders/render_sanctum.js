// scripts/renders/render_sanctum.js
function renderSanctuaryScene(container, frame) {
    container.innerHTML = '';

    // Título
    const titleDiv = document.createElement('div');
    titleDiv.className = 'room-title-display';
    titleDiv.innerHTML = '<span style="color:var(--accent-blood)">SANCTUM ZONE</span>';
    container.appendChild(titleDiv);

    // Wrapper
    const sanctumWrapper = document.createElement('div');
    sanctumWrapper.className = 'sanctuary-container';

    // Camadas
    const svgLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgLayer.classList.add("sanctuary-svg");
    sanctumWrapper.appendChild(svgLayer);

    const gridLayer = document.createElement('div');
    gridLayer.className = 'sanctuary-grid';    
    
    const nodeOccupancy = {}; 
    const itemsOnFloor = {}; 
    let edgesToDraw = [];

    // Identifica a Câmera
    const cameraAgentId = window.selectedAgentId || Object.keys(frame.agents)[0];
    const cameraAgent = frame.agents[cameraAgentId];

    // Busca o Mapa (Arestas) em QUALQUER agente    
    Object.values(frame.agents).forEach(ag => {
        if (ag.arena_edges && ag.arena_edges.length > 0) {
            edgesToDraw = ag.arena_edges;
        }
    });

    // Define quem desenhar (Lista VIP ou Todos)
    let agentsToDraw = [];
    
    // Tenta modo restrito (Eu + Oponente)
    if (cameraAgent && cameraAgent.opponent_id) {
        agentsToDraw.push(cameraAgentId);
        agentsToDraw.push(cameraAgent.opponent_id);
    } else {
        // Fallback: Se não sei quem é o oponente, desenha TODO MUNDO que está na arena
        // Isso evita a tela vazia
        agentsToDraw = Object.keys(frame.agents);
    }

    // Processamento de Posições
    agentsToDraw.forEach(agentId => {
        const ag = frame.agents[agentId];
        if (!ag) return;

        // Verifica se está visualmente na arena (nó k_...)
        const isInSanctumNode = ag.location_node && ag.location_node.startsWith('k_');
        
        // Se não está num nó 'k_', ignora (está em outro lugar/waiting)
        if (!isInSanctumNode) return;

        const nodeIdx = parseKNodeIndex(ag.location_node);
        
        if (nodeIdx !== null) {
            if (!nodeOccupancy[nodeIdx]) nodeOccupancy[nodeIdx] = [];
            nodeOccupancy[nodeIdx].push({...ag, id: agentId});

            if (ag.room_items && ag.room_items.length > 0) {
                itemsOnFloor[nodeIdx] = ag.room_items;
            }
        }
    });

    // Desenha Linhas (Arestas)
    if (edgesToDraw.length > 0) {
        edgesToDraw.forEach(edge => {
            const [u, v] = edge;
            const line = createSvgLine(u, v);
            svgLayer.appendChild(line);
        });
    } else {
        console.warn("Aviso: Nenhuma aresta de arena encontrada no JSON para este frame.");
    }

    // Desenha Nós (0 a 8)
    for (let i = 0; i < 9; i++) {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = `k-node`;
        
        const floorDiv = document.createElement('div');
        floorDiv.className = 'k-room-floor';
        
        // Renderiza Agentes
        if (nodeOccupancy[i]) {
            nodeDiv.classList.add('has-agent');
            
            // Se tem mais de 1 agente, é um encontro -> Brilha
            if (nodeOccupancy[i].length > 1) {
                nodeDiv.classList.add('has-exit'); 
            }

            nodeOccupancy[i].forEach((ag, idx) => {
                const img = document.createElement('img');
                if (window.AssetManager) {
                    img.src = AssetManager.getAgentSprite(ag.name);
                }
                img.className = 'k-avatar';
                
                // Offset visual para não sobrepor bonecos na mesma sala
                if (nodeOccupancy[i].length > 1) {
                    const offset = idx === 0 ? -12 : 12;
                    img.style.transform = `translateX(${offset}px)`;
                    img.style.zIndex = idx + 10;
                    // Borda colorida para diferenciar (Vermelho vs Azul)
                    img.style.borderColor = idx === 0 ? 'var(--accent-soul)' : 'var(--accent-blood)';
                }
                floorDiv.appendChild(img);
            });
        }

        // Renderiza Itens
        if (itemsOnFloor[i]) {
            const itemIcon = document.createElement('div');
            itemIcon.className = 'k-item';
            const isArtifact = itemsOnFloor[i].some(it => it.toLowerCase().includes('amulet') || it.toLowerCase().includes('ring'));
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

// Funções auxiliares
function createSvgLine(u, v) {
    const getCoord = (idx) => {
        const row = Math.floor(idx / 3);
        const col = idx % 3;
        // Coordenadas centrais (em %)
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
    const parts = nodeId.split('_');
    // Formato: k_{floor}_{index}
    return parseInt(parts[parts.length - 1]);
}