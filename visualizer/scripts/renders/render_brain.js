// render_brain.js
function renderNeuralNet(agent) {
    renderRawStates(agent.raw_obs);
    drawNetworkCanvas(agent.raw_obs, agent.action_taken);
}

function renderRawStates(rawObs) {
    const grid = document.getElementById('raw-state-grid');
    if (!grid || !rawObs) return;
    grid.innerHTML = '';
    
    rawObs.forEach((val, idx) => {
        const cell = document.createElement('div');
        cell.className = 'state-cell';
        cell.title = `Input [${idx}]: ${val.toFixed(2)}`;
        if (val > 0.1) {
            cell.classList.add('state-active-pos');
            cell.style.opacity = Math.min(1, val);
        } else if (val < -0.1) {
            cell.classList.add('state-active-neg');
            cell.style.opacity = Math.min(1, Math.abs(val));
        }
        grid.appendChild(cell);
    });
}

function drawNetworkCanvas(inputs, actionName) {
    const canvas = document.getElementById('neural-net-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    const layers = [
        { count: 10, x: 30, color: '#444' },
        { count: 14, x: w/2, color: '#666' },
        { count: 10, x: w-30, color: '#888' }
    ];
    
    const actionMap = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait", "Equip", "Social", "Move N", "Move S", "Move E", "Move W"];
    let actionIndex = -1;
    if (typeof actionName === 'string') {
        actionIndex = actionMap.indexOf(actionName);
        if (actionIndex === -1 && actionName.includes("Move")) actionIndex = 6;
    }

    const getY = (idx, count) => (h / (count + 1)) * (idx + 1);
    ctx.lineWidth = 0.5;
    
    // Conexões
    for (let i = 0; i < layers[0].count; i++) {
        for (let j = 0; j < layers[1].count; j++) {
            const inputVal = inputs ? (inputs[i] || 0) : 0;
            const isActive = Math.abs(inputVal) > 0.5;
            ctx.strokeStyle = isActive ? 'rgba(164, 176, 190, 0.4)' : 'rgba(50,50,50, 0.15)';
            ctx.beginPath(); ctx.moveTo(layers[0].x, getY(i, layers[0].count)); ctx.lineTo(layers[1].x, getY(j, layers[1].count)); ctx.stroke();
        }
    }
    for (let i = 0; i < layers[1].count; i++) {
        for (let j = 0; j < layers[2].count; j++) {
            const isChosenPath = (j === actionIndex);
            ctx.strokeStyle = isChosenPath ? 'rgba(127, 0, 0, 0.9)' : 'rgba(50,50,50, 0.15)';
            ctx.lineWidth = isChosenPath ? 1.5 : 0.5;
            ctx.beginPath(); ctx.moveTo(layers[1].x, getY(i, layers[1].count)); ctx.lineTo(layers[2].x, getY(j, layers[2].count)); ctx.stroke();
            ctx.lineWidth = 0.5;
        }
    }

    // Nós
    layers.forEach((layer, lIdx) => {
        for (let i = 0; i < layer.count; i++) {
            const cx = layer.x, cy = getY(i, layer.count);
            ctx.beginPath();
            let radius = 3, fill = layer.color, stroke = null;

            if (lIdx === 0 && inputs && Math.abs(inputs[i]) > 0.5) {
                fill = inputs[i] > 0 ? '#e74c3c' : '#3498db'; radius = 4;
            } else if (lIdx === 2 && i === actionIndex) {
                fill = '#e74c3c'; stroke = '#fff'; radius = 6;
                ctx.arc(cx, cy, 10, 0, 2*Math.PI); ctx.strokeStyle = 'rgba(231, 76, 60, 0.5)'; ctx.stroke(); ctx.beginPath();
            }
            ctx.arc(cx, cy, radius, 0, 2*Math.PI); ctx.fillStyle = fill; ctx.fill();
            if(stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
        }
    });
}