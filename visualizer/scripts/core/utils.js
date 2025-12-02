// utils.js - Funções auxiliares

function formatRoomName(rawId) {
    if (!rawId) return "UNKNOWN";
    if (rawId.startsWith("p_")) {
        const parts = rawId.split('_');
        return `ZONE ${parts[1]}-${parts[2]}`;
    }
    if (rawId.startsWith("k_")) {
        const parts = rawId.split('_');
        return `ARENA ${parts[1]}-${parts[2]}`;
    }
    if (rawId.toLowerCase() === 'start') return "ENTRY POINT";
    return rawId.toUpperCase();
}

function detectEnemyRank(name) {
    if (!name) return 'blue';
    const n = name.toLowerCase();
    if (n.includes('king') || n.includes('lord') || n.includes('boss')) return 'red';
    if (n.includes('elite') || n.includes('captain') || n.includes('(x)')) return 'yellow';
    return 'blue';
}