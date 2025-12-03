// scripts/main.js

const RECORD_PATH = 'recordings/recording.json'; 
window.replayData = [];
let selectedAgentId = null;

// --- RENDERIZADOR MASTER ---
function renderFrame(index) {
    if (!window.replayData || index >= window.replayData.length) return;
    
    const frame = window.replayData[index];
    const agentId = selectedAgentId || Object.keys(frame.agents)[0];
    const agent = frame.agents[agentId];

    if (!agent) return;

    // 1. Atualiza HUD e UI
    document.getElementById('floor-display').innerText = `FLOOR ${agent.floor}`;
    document.getElementById('turn-display').innerText = frame.turn;
    
    updateStats(agent, agent.action_taken); 
    drawKarma(agent.karma);                 
    updateLogs(agent.logs);                 

    // 2. Atualiza Palco (Roteador de Cenas)
    const stage = document.getElementById('stage');
    
    if (agent.scene_mode === "WAITING") {
        renderWaitingScreen(stage);            
    } else if (agent.scene_mode.includes("COMBAT")) {
        renderCombatScene(stage, agent, index);
    } else if (agent.scene_mode === "ARENA") {
        renderSanctuaryScene(stage, frame);  
    } else {
        renderExplorationScene(stage, agent, index); 
    }

    // 3. Atualiza Brain
    renderNeuralNet(agent);
}

// --- BOOT ---
function initSystem(data) {
    window.replayData = data;
    console.log(`Dados carregados: ${data.length} frames.`);

    // Configura Câmera de acordo com o agente selecionado    
    selectedAgentId = setupCameraControl(data, (newId) => {
        selectedAgentId = newId;
        PlaybackManager.render(); 
    });

    // Inicia PlaybackManager
    const startApp = () => PlaybackManager.init(window.replayData, renderFrame);

    if(window.AssetManager) {
        AssetManager.preloadAll().then(startApp);
    } else {
        console.warn("AssetManager não encontrado.");
        startApp();
    }
}

// Fetch Inicial
fetch(RECORD_PATH).then(r=>r.ok?r.json():null).then(d => {
    if(d) { initSystem(d); }
    else { console.warn("Usando Mock Data"); initSystem(mockReplay); }
}).catch(e => {
    console.error("Erro:", e);
    initSystem(mockReplay);
});