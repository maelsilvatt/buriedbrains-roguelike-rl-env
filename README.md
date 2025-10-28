# BuriedBrains: A Roguelike-Inspired Single-Agent RL Environment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ## 📜 Visão Geral

**BuriedBrains** é um ambiente de simulação procedural, parcialmente observável (POMDP) e de alto risco, projetado como um benchmark para pesquisa em Aprendizado por Reforço (RL), com foco no estudo de agentes com memória e na emergência de comportamentos complexos. Inspirado em jogos do gênero roguelike, o ambiente utiliza mecânicas como morte permanente (`permadeath`) e geração procedural de níveis para criar cenários desafiadores que exigem planejamento estratégico, gerenciamento de risco e adaptação sob incerteza.

Esta versão do repositório (`buriedbrains-roguelike-sae`) foca na **Parte 1** do projeto: o **Ambiente Single-Agent PvE (Player versus Environment)**. O objetivo desta fase é validar o core do ambiente, testar a capacidade de aprendizado de agentes RL (PPO, LSTM) e investigar a necessidade de memória em um POMDP complexo.

## 🎮 Inspiração

O projeto **BuriedBrains** foi inspirado diretamente no jogo mobile *Buriedbornes*, desenvolvido pela Nussygame. Este jogo combina elementos clássicos de roguelikes com combate tático por turnos, progressão baseada em risco e morte permanente — características que influenciaram fortemente o design do ambiente. A simplicidade visual aliada à profundidade estratégica de *Buriedbornes* serviu como base conceitual para criar um ambiente de RL desafiador, parcialmente observável e com geração procedural, ideal para investigar agentes com memória e tomada de decisão sob incerteza.

Este projeto foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação na Universidade Federal do Ceará (UFC) - Campus Sobral.

## ✨ Funcionalidades Principais (Parte 1 - PvE)

* **Ambiente Gymnasium-Compatível:** Interface padrão para fácil integração com frameworks de RL como Stable Baselines3.
* **Geração Procedural Baseada em Grafos:** Os níveis (andares de progressão) são modelados como Grafos Acíclicos Dirigidos (DAGs) gerados dinamicamente, com poda de ramos não escolhidos.
* **Parcial Observabilidade (POMDP):** O agente possui uma visão limitada do ambiente, necessitando de memória ou inferência para tomar decisões ótimas.
* **Combate Tático:** Sistema de combate por turnos com habilidades, cooldowns, efeitos de status e gerenciamento de HP.
* **Progressão e Risco:** Mecânica de morte permanente (`permadeath`) e sistema de níveis/experiência.
* **Conteúdo Configurável via YAML:** Inimigos, habilidades, equipamentos, eventos e efeitos de sala são definidos em arquivos YAML, permitindo fácil balanceamento e extensão.
* **Geração de Conteúdo Baseada em Budget:** A dificuldade e variedade das salas são controladas por um sistema de "orçamento" e regras condicionais.
* **Logging Detalhado e Hall da Fama:** Callback customizado para Stable Baselines3 que registra métricas detalhadas e salva os logs completos das runs mais bem-sucedidas.

## 🎯 Motivação e Objetivos

O objetivo central desta fase do BuriedBrains é fornecer um benchmark robusto para investigar questões fundamentais em IA:

* **Necessidade de Memória:** Testar experimentalmente como a capacidade de memória (e.g., LSTM) impacta o desempenho em ambientes POMDP com desafios sequenciais e mecânicas complexas (Hipótese H1).
* **Tomada de Decisão sob Risco:** Analisar como a mecânica de `permadeath` influencia o desenvolvimento de estratégias prudentes versus agressivas (Hipótese H2).
* **Generalização:** Avaliar se os agentes aprendem políticas generalizáveis que funcionam em níveis gerados proceduralmente, em vez de memorizar soluções específicas (Hipótese H4).

## 📊 Status Atual e Resultados Principais

* O ambiente PvE single-agent está funcional e passou por vários ciclos de balanceamento.
* Experimentos comparando PPO (sem memória) e RecurrentPPO (LSTM) demonstraram que, **no ambiente atual com chefes e mecânicas complexas, a memória (LSTM) é crucial para o aprendizado e a sobrevivência**, validando a Hipótese H1 para este cenário.
* O agente LSTM é capaz de aprender políticas para sobreviver e progredir no ambiente, embora a duração longa dos combates contra chefes seja um gargalo para completar o jogo consistentemente dentro do limite de tempo padrão (30k passos).

## 🛠️ Arquitetura e Tecnologias

* **Linguagem:** Python 3.x
* **Core RL:** Gymnasium, Stable Baselines3
* **Computação Numérica:** NumPy, PyTorch (via Stable Baselines3)
* **Grafos:** NetworkX (para modelagem e manipulação da topologia)
* **Configuração:** PyYAML

## 🚀 Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/maelsilvatt/buriedbrains-roguelike-rl-env.git](https://github.com/maelsilvatt/buriedbrains-roguelike-rl-env.git)
    cd buriedbrains-roguelike-rl-env
    ```
2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # ou
    .\venv\Scripts\activate  # Windows
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Uso (Treinamento)

O script principal para treinar um agente é `train.py`. Ele aceita argumentos de linha de comando para configurar o experimento:

```bash
python train.py [opções]
````

**Opções Principais:**

  * `--no_lstm`: Usa PPO padrão (MlpPolicy) em vez de RecurrentPPO (MlpLstmPolicy).
  * `--total_timesteps <int>`: Número total de passos de treinamento (padrão: 5,000,000).
  * `--max_episode_steps <int>`: Limite de passos por episódio no ambiente (padrão: 30,000).
  * `--budget_multiplier <float>`: Multiplicador de dificuldade (padrão: 1.0). Afeta o "orçamento" para geração de conteúdo.
  * `--suffix <str>`: Adiciona um sufixo customizado ao nome da pasta de log/modelo.

**Exemplos:**

  * **Treinar LSTM (padrão) por 1M de passos:**
    ```bash
    python train.py --total_timesteps 1000000 --suffix "LSTM_Test_1M"
    ```
  * **Treinar PPO (sem LSTM) por 5M de passos com limite de 50k por episódio:**
    ```bash
    python train.py --no_lstm --total_timesteps 5000000 --max_episode_steps 50000 --suffix "PPO_Baseline_5M_50kSteps"
    ```

Os logs do TensorBoard serão salvos na pasta `logs/` e os modelos e Hall da Fama na subpasta correspondente dentro de `models/` e `logs/`.

## 🔮 Trabalhos Futuros

Embora este repositório foque na Parte 1 (PvE), o design completo do BuriedBrains prevê uma **Parte 2** focada em interações **Multiagente (MAE)** e **Comportamento Social Emergente**:

  * Implementação das "Zonas K" com topologia de grafo não-direcionado para encontros PvP.
  * Introdução de ações sociais (e.g., Soltar/Pegar Artefato para barganha inferida).
  * Implementação do sistema de Karma para rastrear reputação e influenciar interações.
  * Refatoração do ambiente para a API multiagente.
  * Desenvolvimento de um loop de treinamento MARL (provavelmente Self-Play)
  * Validação da Hipótese sobre a emergência de comportamentos sociais contextuais.
  * Desenvolvimento completo do Visualizador externo em Unity.

## 📄 Citação

Se usar este ambiente em sua pesquisa, por favor, cite o trabalho
```bibtex
@misc{silva2025buriedbrains,
  author = {Silva, Ismael Soares da},
  title = {BuriedBrains: BuriedBrains: Um Ambiente Roguelike Parcialmente Observável para Benchmark de Agentes RL com Memória},
  year = {2025},
  howpublished = {Trabalho de Conclusão de Curso (Engenharia da Computação), Universidade Federal do Ceará, Campus Sobral},
  note = {Orientador: Prof. Dr. Thiago Iachiley Araújo de Souza}
}
```

## ⚖️ Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

```