# # Importando as funções de centralidade que você já modelou
# from .centrality_metrics import calculate_betweenness, calculate_degree

# def gamma_generate_content(vertex_v, floor_k, graph_G, seed):
#     """
#     Função estocástica para geração de conteúdo de uma sala.
#     """
#     # Define a semente para reprodutibilidade
#     random.seed(seed)

#     # 1. Calcular o orçamento de dificuldade
#     base_budget = 10 * (floor_k ** 1.1)
    
#     # Ajustar orçamento com base na importância tática da sala (centralidade)
#     # [cite: 193, 195]
#     betweenness_centrality = calculate_betweenness(vertex_v, graph_G)
#     final_budget = base_budget * (1 + 0.5 * betweenness_centrality)

#     # Inicializar o vetor de atributos da sala [cite: 231]
#     room_attributes = {'enemies': [], 'items': [], 'effects': []}
    
#     # 2. Definir custos e pools de conteúdo
#     # 
#     content_costs = {
#         'Inimigo Comum': 10, 'Inimigo Forte (H)': 25, 'Inimigo de Elite (X)': 50,
#         'Armadilha de Dano': 8, 'Tesouro Protegido (T)': -30, 'Fonte de Cura': -20
#     }
    
#     # Selecionar o pool de conteúdo elegível com base no andar 'k'
#     eligible_content = get_eligible_pool(floor_k)

#     # 3. "Gastar" o orçamento
#     while final_budget > 0 and len(eligible_content) > 0:
#         # Escolher um item do pool elegível de forma probabilística
#         # (pode dar mais chance a inimigos do que a armadilhas, por exemplo)
#         chosen_content_name = random.choices(
#             population=list(eligible_content.keys()), 
#             weights=list(eligible_content.values()), 
#             k=1
#         )[0]
        
#         cost = content_costs[chosen_content_name]

#         # Se houver orçamento, adiciona o conteúdo e deduz o custo
#         if final_budget - cost >= 0:
#             # Adicionar ao vetor de atributos da sala
#             add_content_to_attributes(room_attributes, chosen_content_name)
#             final_budget -= cost
#         else:
#             # Se não há mais orçamento para este item, remove-o das opções
#             eligible_content.pop(chosen_content_name)

#     return room_attributes