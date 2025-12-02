// scripts/render_waiting.js
function renderWaitingScreen(container) {
    container.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#aaa; animation: pulse 2s infinite;">
            <i class="fa-solid fa-hourglass-half fa-spin" style="font-size:50px; margin-bottom:20px; color:var(--accent-soul);"></i>
            <h2 style="font-family:'Press Start 2P'; font-size:14px; margin-bottom:10px;">AGUARDANDO...</h2>
            <p style="font-family:'Crimson Pro'; font-size:16px;">Agente chegou ao Santuário.</p>
        </div>
    `;
}