@echo OFF
call corevenv\Scripts\activate.bat
echo [INICIANDO SCRIPT DE TREINAMENTO EM OFF]
echo O primeiro treino (5M steps) pode levar mais de 4 horas.
echo Os treinos seguintes (2M steps) devem levar aprox. 1.5-2h cada.
echo.
:: --- Experimento 1: Baseline PPO (Com LSTM) ---
python train.py --total_timesteps 5000000 --max_episode_steps 30000 --budget_multiplier 1.0 --suffix "PPO_Baseline_5M"
echo =================================================================
echo [RUN 1/5] Iniciando Baseline PPO (Sem LSTM) - 5M Steps...
echo OBJETIVO: Provar que PPO sem memoria e pior (grupo de controle).
echo.
:: --- Experimento 1: Baseline PPO (Sem LSTM) ---
python train.py --no_lstm --total_timesteps 5000000 --max_episode_steps 30000 --budget_multiplier 1.0 --suffix "PPO_Baseline_5M"
echo.
echo [RUN 1/5] Baseline PPO CONCLUIDO.
echo =================================================================
echo.


echo =================================================================
echo [RUN 2/5] Iniciando LSTM com Tempo Limite Estendido (50k) - 2M Steps...
echo OBJETIVO: Testar se o agente vence o jogo se tiver mais tempo.
echo.
:: --- Experimento 2: LSTM com mais tempo limite ---
python train.py --total_timesteps 2000000 --max_episode_steps 50000 --budget_multiplier 1.0 --suffix "LSTM_MaxSteps_50k"
echo.
echo [RUN 2/5] Teste de Tempo Limite CONCLUIDO.
echo =================================================================
echo.


echo =================================================================
echo [RUN 3/5] Iniciando LSTM Modo Facil (Budget 0.8) - 2M Steps...
echo OBJETIVO: Testar se o agente vence o jogo se os inimigos forem 20% mais fracos.
echo.
:: --- Experimento 3: LSTM Modo Facil ---
python train.py --total_timesteps 2000000 --max_episode_steps 30000 --budget_multiplier 0.8 --suffix "LSTM_EasyMode_0.8"
echo.
echo [RUN 3/5] Teste de Modo Facil CONCLUIDO.
echo =================================================================
echo.


echo =================================================================
echo [RUN 4/5] Iniciando LSTM Modo DIFICIL (Budget 1.2) - 2M Steps...
echo OBJETIVO: Testar se o agente aprende a ser mais forte com inimigos 20% mais dificeis.
echo.
:: --- Experimento 4: LSTM Modo Dificil ---
python train.py --total_timesteps 2000000 --max_episode_steps 30000 --budget_multiplier 1.2 --suffix "LSTM_HardMode_1.2"
echo.
echo [RUN 4/5] Teste de Modo Dificil CONCLUIDO.
echo =================================================================
echo.


echo =================================================================
echo [RUN 5/5] Iniciando LSTM Modo PRESSAO (Steps 15k) - 2M Steps...
echo OBJETIVO: Testar se o agente aprende a ser mais eficiente (lutas rapidas) com metade do tempo.
echo.
:: --- Experimento 5: LSTM Modo Pressao ---
python train.py --total_timesteps 2000000 --max_episode_steps 15000 --budget_multiplier 1.0 --suffix "LSTM_PressureMode_15k"
echo.
echo [RUN 5/5] Teste de Modo Pressao CONCLUIDO.
echo =================================================================
echo.


echo [TODOS OS 5 TREINAMENTOS EM OFF FORAM CONCLUIDOS]
pause