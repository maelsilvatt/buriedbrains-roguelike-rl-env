# BuriedBrains: A Roguelike-Inspired Multi-Agent RL Environment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18079360.svg)](https://doi.org/10.5281/zenodo.18079360)

## 📜 Visão Geral

**BuriedBrains** é um ambiente de simulação procedural, parcialmente observável (POMDP) e de alto risco, projetado como um benchmark para pesquisa em Aprendizado por Reforço (RL). O projeto evoluiu de um ambiente *Single-Agent* (Fase 1) para uma arquitetura **Multi-Agent (Fase 2)** completa, capaz de suportar interações sociais complexas, combate PvP e dilemas de cooperação versus traição.

Inspirado em jogos do gênero roguelike, o ambiente utiliza mecânicas como morte permanente (com respawn estratégico), geração procedural de níveis baseada em grafos e um sistema de reputação **Karma** persistente.

Este repositório contém a implementação completa do ambiente, os scripts de treinamento e as ferramentas de validação utilizadas no Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação na Universidade Federal do Ceará (UFC) - Campus Sobral.

## 🎮 Inspiração

O projeto **BuriedBrains** foi inspirado diretamente no jogo mobile *Buriedbornes* (Nussygame). A simplicidade visual aliada à profundidade estratégica desse jogo serviu como base para criar um ambiente onde agentes devem gerenciar cooldowns, equipamentos e riscos. Na Fase 2, o projeto expande esse conceito introduzindo "Zonas de Encontro" (Santuários), inspiradas em lobbies multiplayer e dilemas sociais da Teoria dos Jogos.

## ✨ Funcionalidades Principais

### Fase 1: Core PvE (Validado)
* **Geração Procedural via Grafos:** Níveis de progressão modelados como Grafos Acíclicos Dirigidos (DAGs) com poda dinâmica de ramos não escolhidos.
* **Parcial Observabilidade (POMDP):** O agente opera com um vetor de observação limitado (38 estados), exigindo memória (LSTM) para inferir contextos táticos.
* **Combate Tático:** Sistema de turnos com skills, cooldowns, efeitos de status (Stun, DoT, Buffs) e escalonamento de dificuldade.
* **Sistema de Equipamentos:** Loot com raridade (Comum a Lendário) e lógica de decisão estratégica para upgrades.

### Fase 2: Arquitetura Social & Multiagente (Implementada)
* **Estrutura MAE (Multi-Agent Environment):** O ambiente gerencia múltiplos agentes simultâneos com espaços de ação/observação independentes (`gym.spaces.Dict`).
* **Máquina de Estados Híbrida:** * `PROGRESSION`: Agentes exploram seus próprios mundos PvE isolados.
    * `ARENA_SYNC`: Mecânica de sincronização temporal para aguardar oponentes.
    * `ARENA_INTERACTION`: Transição para grafos cíclicos (`Erdős-Rényi`) onde agentes interagem fisicamente.
* **Mecânicas Sociais:**
    * **Barganha Inferida:** Detecção de intenção cooperativa através de ações de "Dropar/Pegar Artefato".
    * **Traição:** Detecção de ataques após ofertas de paz, com penalidades severas de Karma.
    * **Sistema de Karma:** Modelo de reputação persistente que sobrevive à morte do agente, permitindo consequências de longo prazo em jogos iterados.
    * **Morte e Respawn:** Agentes derrotados reiniciam sua progressão (Nível 1), mas mantêm sua identidade e histórico social (Karma).

## 🎯 Objetivos Científicos

Este projeto foi desenhado para testar hipóteses específicas sobre Inteligência Artificial:

1. **Relevância da Memória (H1):** Validada. Experimentos demonstraram que agentes com memória (LSTM) superam significativamente agentes reativos (PPO) em cenários com chefes e mecânicas temporais complexas.
2. **Tomada de Decisão sob Risco (H2):** Validada. Agentes aprendem a evitar ações inválidas e gerenciar equipamentos para maximizar a sobrevivência.
3. **Emergência de Comportamento Social (H3):** Arquitetura implementada para permitir que estratégias de cooperação ou traição surjam organicamente em função do Karma e do contexto (diferença de poder).

## 📊 Resultados (Fase 1)

* **Validação do Ambiente:** O ambiente provou-se desafiador, com agentes "médios" morrendo no mid-game (andares 100-150) devido ao escalonamento de dificuldade.
* **Memória vs. Reativo:** Agentes LSTM demonstraram estabilidade de aprendizado (`explained_variance` ~0.7), enquanto agentes PPO sofreram colapso de política em cenários complexos.
* **Estratégia de Equipamento:** Logs comprovam que o agente aprendeu a comparar a raridade de itens no chão com os equipados, realizando apenas trocas vantajosas.

## 🛠️ Tecnologias

* **Linguagem:** Python 3.x
* **Core RL:** Gymnasium, Stable Baselines3, SB3-Contrib (RecurrentPPO)
* **Otimização:** Optuna (Hyperparameter Optimization)
* **Grafos:** NetworkX (Modelagem topológica de Dungeons e Arenas)
* **Configuração:** PyYAML
* **Dados:** NumPy, PyTorch

## 🚀 Instalação

1. **Clone o repositório:**
    ```bash
    git clone [https://github.com/maelsilvatt/buriedbrains-roguelike-rl-env.git](https://github.com/maelsilvatt/buriedbrains-roguelike-rl-env.git)
    cd buriedbrains-roguelike-rl-env
    ```
2. **Crie o ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # ou
    .\venv\Scripts\activate  # Windows
    ```
3. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Uso (Treinamento)

O script `train.py` suporta treinamento de longa duração, checkpoints e continuação de treino (resume).

```bash
python train.py [opções]

```

**Opções Principais:**

* `--total_timesteps <int>`: Total de passos de treino (ex: 5000000).
* `--max_episode_steps <int>`: Duração máxima do episódio/vida (recomendado: 50000 para Fase 2).
* `--budget_multiplier <float>`: Dificuldade do gerador de conteúdo (1.0 = Normal).
* `--load_path <str>`: Caminho para um arquivo `.zip` de modelo para **continuar o treinamento**.
* `--suffix <str>`: Nome identificador da run (para logs e salvamento).

## 🔮 Próximos Passos (Fase 3 - Experimentos Sociais)

Com a arquitetura MAE implementada no `env.py`, os próximos passos da pesquisa envolvem:

* **Treinamento Self-Play:** Implementar um loop de treino customizado para alimentar a rede neural com as experiências de ambos os agentes (`a1`, `a2`) simultaneamente.
* **Análise de Karma:** Executar simulações de longa duração para observar se o Karma acumulado influencia a taxa de agressão em encontros futuros (vingança/cooperação).
* **Visualização:** Conectar o simulador a uma interface gráfica (Unity) via sockets para demonstrar as interações em tempo real.

## 📄 Citação

Se este software foi útil para sua pesquisa, por favor cite-o conforme abaixo:

```bibtex
@misc{silva2025buriedbrains,
  author       = {Silva, Ismael Soares da},
  title        = {BuriedBrains: Um Ambiente Multiagente Procedural e Parcialmente Observável para Benchmark de Memória},
  year         = {2025},
  version      = {v2.3.2},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18079360},
  howpublished = {\url{[https://doi.org/10.5281/zenodo.18079360](https://doi.org/10.5281/zenodo.18079360)}},
  note         = {Trabalho de Conclusão de Curso (Engenharia da Computação) -- Universidade Federal do Ceará, Campus Sobral. Orientador: Prof. Dr. Thiago Iachiley Araújo de Souza}
}

```

## ⚖️ Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

```
