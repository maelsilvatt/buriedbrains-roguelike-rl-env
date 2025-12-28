# BuriedBrains: Um Ambiente Multiagente Procedural e Parcialmente Observável para Benchmark de Memória

## 📜 Visão Geral

O **BuriedBrains** é um benchmark experimental desenvolvido para isolar e mensurar a capacidade de planejamento estratégico e persistência temporal de agentes de Aprendizado por Reforço (RL). Desenvolvido como Trabalho de Conclusão de Curso em Engenharia da Computação na **Universidade Federal do Ceará (UFC) - Campus Sobral** , o ambiente utiliza a estrutura rigorosa do gênero *roguelike* para desafiar a generalização em Processos de Decisão de Markov Parcialmente Observáveis (POMDPs).

O ambiente caracteriza-se por:

* 
**Geração Procedural (PCG):** Heurísticas abstratas baseadas em orçamento (*Budget-Based Generation*).


* 
**Observabilidade Parcial:** Exige manutenção de estado interno para lidar com informações ocultas.


* 
**Alta Pressão:** Condições de *permadeath* e escassez de recursos que atuam como gargalos matemáticos para políticas reativas.



## 🏗️ Arquitetura do Simulador

O simulador adota uma estrutura modular **Hub-and-Spoke**, garantindo escalabilidade e separação de preocupações.

### Topologia Híbrida de Grafos

O ambiente alterna entre dois paradigmas topológicos para induzir diferentes modos de raciocínio:

* 
**Zonas de Progressão (PvE):** Modeladas como **Grafos Acíclicos Dirigidos (DAGs)**, onde cada bifurcação representa uma decisão irreversível e custo de oportunidade.


* 
**Santuários (Arenas PvP):** Grafos não-dirigidos cíclicos baseados no modelo **Erdős-Rényi**, otimizados via poda baseada em centralidade para criar pontos de estrangulamento (*chokepoints*).



### Sistema de Reputação Hiperbólica (Karma)

A confiança e a moralidade dos agentes são mapeadas no interior de um **Disco de Poincaré**. A evolução do Karma segue uma Equação Diferencial Estocástica (SDE), onde ações benevolentes ou malévolas deslocam o estado moral em direção a polos magnéticos ("Santo" vs. "Vilão").

## 🧠 Camada de Inteligência Artificial

O projeto utiliza um extrator de características por atenção (**Self-Attention**) de 198 dimensões para processar blocos semânticos de habilidades, inventário e sensores sociais.

* 
**Espaço de Observação ():** Estruturado em 11 tokens (Habilidades, Propriocepção, Contexto PvE, Sensores Sociais, Navegação, etc.).


* 
**Espaço de Ação ():** Inclui ativação de habilidades, interação, movimento, uso de consumíveis e sinalização social por descarte de itens (*Drop*).


* 
**Modelos Comparativos:** O benchmark contrasta uma arquitetura reativa (**PPO**) com uma recorrente (**LSTM/RecurrentPPO**) para isolar o impacto da memória na resolução de problemas de longo prazo.



## 📉 Dinâmica de Sobrevivência e "Chaos Mode"

Para evitar a estagnação e forçar a eficiência, o ambiente impõe:

* 
**Escalonamento Logístico:** HP e Dano dos inimigos crescem via curva sigmoide até o andar 500.


* 
**Chaos Mode:** Após o andar 500, o crescimento torna-se exponencial (), testando o limite máximo de generalização.


* 
**Floor Tax:** Dano fixo por andar () que ignora parcialmente a defesa, garantindo desgaste contínuo.



## 🛠️ Tecnologias e Infraestrutura

* 
**Core:** Python 3.14, Gymnasium, Stable Baselines3.


* 
**Redes Neurais:** PyTorch (Self-Attention & LSTM).


* 
**Grafos:** NetworkX.


* 
**Visualizer:** Interface Web (JavaScript/HTML5) para análise qualitativa e telemetria neural em tempo real.



## 🚀 Instalação e Uso

1. **Clone o repositório:**
```bash
git clone https://github.com/maelsilvatt/buriedbrains-roguelike-rl-env.git
cd buriedbrains-roguelike-rl-env

```


2. **Instale as dependências:**
```bash
pip install -r requirements.txt

```


3. **Treinamento:**
Execute os scripts de treinamento (ex: `Magnolia`, `Be'helit`) documentados no Apêndice A para reproduzir os experimentos.



## 📄 Citação

Este trabalho foi desenvolvido por **Ismael Soares da Silva** sob orientação do Prof. Dr. Thiago Iachiley Araújo de Souza.

```bibtex
@misc{silva2025buriedbrains,
  author       = {Silva, Ismael Soares da},
  title        = {BuriedBrains: Um Ambiente Multiagente Procedural e Parcialmente Observável para Benchmark de Memória},
  year         = {2025},
  version      = {v2.3.2},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18079360},
  howpublished = {Trabalho de Conclusão de Curso (Engenharia da Computação) -- Universidade Federal do Ceará, Campus Sobral},
  note         = {Orientador: Prof. Dr. Thiago Iachiley Araújo de Souza. Disponível em: \url{https://doi.org/10.5281/zenodo.18079360}}
}

```
## ⚖️ Licença

Este projeto é licenciado sob a **Licença MIT**. Veja o arquivo `LICENSE` para mais detalhes.
