// data.js - Dados estáticos e Mock

const EFFECT_LIBRARY = {
    'Slow Terrain': { desc: "O terreno difícil aplica 'Slow' em todos no início do combate.", type: 'debuff', icon: 'fa-shoe-prints' },
    'Heat': { desc: "O calor intenso aplica 'Burn' em todos a cada rodada.", type: 'debuff', icon: 'fa-fire' },
    'Dense Fog': { desc: "A neblina densa aplica 'Blind' em todos no início do combate.", type: 'debuff', icon: 'fa-smog' },
    'Weakening Field': { desc: "Uma aura profana reduz a eficácia de curas em 50% na sala.", type: 'debuff', icon: 'fa-heart-crack' },
    'Evasion Zone': { desc: "Uma névoa mágica concede bônus de evasão a todos.", type: 'buff', icon: 'fa-wind' },
    'Amplifying Field': { desc: "O campo de energia aumenta o dano de todos em 15%.", type: 'buff', icon: 'fa-hand-fist' },
    'Consecrated Ground': { desc: "O solo sagrado cura todos os combatentes em 2% HP a cada rodada.", type: 'buff', icon: 'fa-notes-medical' },
    'Conductive Field': { desc: "A energia arcana aumenta dano mágico causado e recebido em 25%.", type: 'neutral', icon: 'fa-bolt' }
};

// Mock de segurança caso não ache o arquivo
const mockReplay = [
    // FRAME 0: Início
    {
        turn: 1,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0, imag: 0},
                location_node: "Start", floor: 1, scene_mode: "EXPLORATION",
                logs: ["Início da Jornada."],
                neighbors: [
                    {id: "p_1_0", has_enemy: true, enemy_type: "Goblin Scout", has_treasure: false},
                    {id: "p_1_1", has_enemy: false, has_treasure: true},
                    {id: "p_1_2", has_enemy: false, has_treasure: false, effect: "Heat Wave"}
                ]
            }
        }
    },
    // FRAME 1: Encontro com Inimigo
    {
        turn: 2,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 100, max_hp: 100, level: 1, karma: {real: 0, imag: 0},
                location_node: "p_1_0", floor: 1, scene_mode: "COMBAT_PVE", // <--- Muda para combate
                combat_enemy: "Goblin Scout",
                logs: ["Entrou na sala p_1_0.", "Encontrou Goblin Scout!"],
                neighbors: []
            }
        }
    },
    // FRAME 2: Tomando Dano (Para testar o Shake)
    {
        turn: 3,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 90, max_hp: 100, level: 1, karma: {real: -0.1, imag: 0.1},
                location_node: "p_1_0", floor: 1, scene_mode: "COMBAT_PVE",
                combat_enemy: "Goblin Scout",
                logs: ["Sofreu 10 de dano."],
                neighbors: []
            }
        }
    },
    // FRAME 3: Vitória e Drop
    {
        turn: 4,
        agents: {
            "a1": {
                name: "Ren_Warden", hp: 90, max_hp: 100, level: 2, karma: {real: -0.1, imag: 0.1}, // Level UP
                location_node: "p_1_0", floor: 1, scene_mode: "EXPLORATION",
                logs: ["Inimigo derrotado!", "Level Up!", "Item encontrado: Iron Sword"],
                neighbors: [
                    {id: "p_2_0", has_enemy: true, enemy_type: "Elite Knight (X)", has_treasure: true} // Teste do Elite
                ]
            }
        }
    }
];