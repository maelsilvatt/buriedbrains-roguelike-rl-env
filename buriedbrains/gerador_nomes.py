import random
import time


class GeradorNomes:
    """
    Gera nomes únicos e variados para agentes de um jogo roguelike ou simulação.
    Garante unicidade dentro da mesma instância.
    """

    def __init__(self, seed=None):
        self.random = random.Random(seed or time.time())
        self.nomes_gerados = set()

        self.nomes_pessoas = [
            "Lucas", "Mateus", "Joao", "Pedro", "Rafael", "Bruno", "Thiago", "Gustavo", "Felipe", "Daniel",
            "Ana", "Julia", "Marina", "Beatriz", "Camila", "Lara", "Isabela", "Carla", "Luana", "Fernanda",
            "Kael", "Zara", "Milo", "Orion", "Sora", "Elara", "Jax", "Kai", "Ren", "Anya",
            "Ryu", "Kenji", "Akira", "Yuki", "Hana", "Talia", "Niko", "Leon", "Max", "Eva"
        ]

        self.apelidos_gamer = [
            "Shadow", "Neo", "Dark", "Cyber", "Ghost", "Iron", "Alpha", "Omega", "Blade", "Storm",
            "Sniper", "Hunter", "Rogue", "Wizard", "Titan", "Viper", "Ninja", "Specter", "Drifter", "Blitz",
            "Reaper", "Slayer", "Warden", "Echo", "Vector", "Havoc", "Fury", "Razor", "Psyche", "Wraith",
            "Jester", "Zero", "Bolt", "Spike", "Hex"
        ]

        self.criaturas = [
            "Wolf", "Dragon", "Phoenix", "Kraken", "Griffin", "Hydra", "Leviathan", "Falcon", "Cobra", "Bear",
            "Wyvern", "Golem", "Behemoth", "Manticore", "Basilisk", "Juggernaut", "Serpent", "Gorgon", "Chimera", "Droid"
        ]

        self.sufixos_tech = [
            "X", "Z", "99", "777", "Prime", "Zero", "One", "MK", "RX", "EXE", "VX", "Ultra", "Core", "Void",
            "Matrix", "Data", "Net", "Sys", "Bot", "Log", "Unit"
        ]

        self.prefixos_tech = [
            "Cyber", "Robo", "Mecha", "Nano", "Bio", "Psy", "Gen", "Xeno", "Proto", "Hyper", "Giga", "Auto"
        ]

        self.titulos_fantasy = [
            "the_Silent", "the_Brave", "of_the_Void", "the_Wanderer", "the_Chosen", "Bloodhand",
            "Ironfist", "Shadowstep", "Stormcaller", "Fireheart", "Ghostblade"
        ]

        self.sufixos_fantasy = [
            "bane", "heart", "soul", "reaver", "shard", "wind", "fury", "shade"
        ]

    def _get_parts(self):
        return {
            "nome": self.random.choice(self.nomes_pessoas),
            "apelido": self.random.choice(self.apelidos_gamer),
            "criatura": self.random.choice(self.criaturas),
            "sufixo_t": self.random.choice(self.sufixos_tech),
            "prefixo_t": self.random.choice(self.prefixos_tech),
            "titulo_f": self.random.choice(self.titulos_fantasy),
            "sufixo_f": self.random.choice(self.sufixos_fantasy),
            "num": self.random.choice(["7", "9", "X", "01", "007", "66", "88", "42"])
        }

    def gerar_nome(self):
        parts = self._get_parts()

        formatos = [
            lambda p: f"{p['apelido']}{p['criatura']}",
            lambda p: f"{p['prefixo_t']}{p['apelido']}",
            lambda p: f"{p['prefixo_t']}{p['criatura']}",
            lambda p: f"{p['apelido']}{p['num']}",
            lambda p: f"{p['apelido']}_{p['sufixo_t']}",
            lambda p: f"{p['criatura']}{p['sufixo_t']}",
            lambda p: f"{p['apelido']}{p['criatura']}{p['sufixo_t']}",
            lambda p: f"{p['nome']}_{p['apelido']}",
            lambda p: f"{p['nome']}_{p['titulo_f']}",
            lambda p: f"{p['nome']}{p['sufixo_f']}",
            lambda p: f"{p['apelido']}",
            lambda p: f"{p['nome']}"
        ]

        base_name = self.random.choice(formatos)(parts)
        final_name = f"{base_name}_{self.random.randint(1, 999)}"

        # garante unicidade local
        while final_name in self.nomes_gerados:
            final_name = f"{base_name}_{self.random.randint(1, 999)}"

        self.nomes_gerados.add(final_name)
        return final_name


if __name__ == "__main__":
    # Teste rápido
    g = GeradorNomes()
    for _ in range(5):
        print(g.gerar_nome())
