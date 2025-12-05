// scripts/ui_controls.js
const PlaybackManager = {
    // Estado
    currentFrame: 0,
    isPlaying: false,
    playInterval: null,
    speedIndex: 0,
    SPEEDS: [1, 1.5, 2, 4],
    baseDelay: 1000,
    
    // Referência aos dados e função de desenho
    data: [],
    renderFn: null, // Será a função renderFrame do main.js

    // Inicialização
    init: function(data, renderFunction) {
        this.data = data;
        this.renderFn = renderFunction;
        this.currentFrame = 0;
        this.isPlaying = false;
        
        // Configura slider
        const timeline = document.getElementById('timeline');
        timeline.min = 0;
        timeline.max = this.data.length - 1;
        timeline.value = 0;

        // Configura Listeners
        this.setupListeners();

        // Renderiza frame 0
        this.render();
    },

    // Controle de velocidade
    getDelay: function() {
        return this.baseDelay / this.SPEEDS[this.speedIndex];
    },

    toggleSpeed: function() {
        this.speedIndex = (this.speedIndex + 1) % this.SPEEDS.length;
        const btn = document.getElementById('btn-speed');
        btn.innerText = `${this.SPEEDS[this.speedIndex]}x`;
        
        if (this.isPlaying) {
            clearInterval(this.playInterval);
            this.playInterval = setInterval(() => this.step(1), this.getDelay());
        }
    },

    play: function() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        
        // Ícone de Pause
        document.getElementById('btn-play').innerHTML = '<i class="fa-solid fa-pause" style="color:var(--accent-blood)"></i>';        
        
        this.playInterval = setInterval(() => this.step(1), this.getDelay());
    },

    pause: function() {
        this.isPlaying = false;
        clearInterval(this.playInterval);
        // Ícone de Play
        document.getElementById('btn-play').innerHTML = '<i class="fa-solid fa-play"></i>';
    },

    step: function(dir) {
        const next = this.currentFrame + dir;
        if (next >= 0 && next < this.data.length) {
            this.currentFrame = next;
            this.render();
        } else {
            this.pause();
        }
    },

    // Renderização interna
    render: function() {
        // Atualiza a barra visualmente
        document.getElementById('timeline').value = this.currentFrame;
        // Chama o main.js para desenhar a tela
        if (this.renderFn) this.renderFn(this.currentFrame);
    },

    // Gatilhos
    setupListeners: function() {        
        document.getElementById('btn-play').onclick = () => this.isPlaying ? this.pause() : this.play();
        document.getElementById('btn-pause').onclick = () => this.pause();
        document.getElementById('btn-next').onclick = () => { this.pause(); this.step(1); };
        document.getElementById('btn-prev').onclick = () => { this.pause(); this.step(-1); };
        document.getElementById('btn-speed').onclick = () => this.toggleSpeed();
        
        document.getElementById('timeline').oninput = (e) => {
            this.pause();
            this.currentFrame = parseInt(e.target.value);
            this.render();
        };
    }
};