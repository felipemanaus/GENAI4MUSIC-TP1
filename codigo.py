import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# ==========================================
# PARÂMETROS GLOBAIS E CONFIGURAÇÕES
# ==========================================
FS = 10                           # Taxa de amostragem: 10 colunas por segundo (resolução de 100ms)
DURACAO_LOOP = 4                  # Duração do loop em segundos (padrão comum em EDM/French House)
TAMANHO_LOOP = DURACAO_LOOP * FS  # Total de colunas na matriz (40 colunas)

# Hiperparâmetros do Algoritmo Genético
POP_SIZE = 60                     # Tamanho da população
NGEN = 50                         # Número total de gerações
CXPB = 0.6                        # Probabilidade de Crossover (60%)
MUTPB = 0.8                       # Probabilidade de Mutação (80% - alta, mas segura devido aos operadores restritos)

# Marcos evolutivos (frames) que serão extraídos para montar a transição gradual[cite: 3]
CHECKPOINTS = [5, 12, 20, 30, 40, 50]

ARQUIVO_MUSICA_A = "Daft Punk — End of Line.mid"
ARQUIVO_MUSICA_B = "Daft Punk — Derezzed.mid"

# ==========================================
# FUNÇÕES DE MANIPULAÇÃO MIDI E PIANO-ROLL
# ==========================================
def extrair_piano_roll_mesclado(caminho_midi, inicio_seg, fim_seg):
    """
    Lê um arquivo MIDI, mescla todos os instrumentos não-percussivos numa única matriz 2D (piano-roll),
    e recorta o trecho temporal especificado.
    """
    midi_data = pretty_midi.PrettyMIDI(caminho_midi)
    
    tempo_final = fim_seg if fim_seg > 0 else midi_data.get_end_time()
    tamanho_total_colunas = int(round(tempo_final * FS)) + 1
    
    pr_mestre = np.zeros((128, tamanho_total_colunas))
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            pr = inst.get_piano_roll(fs=FS)
            limite = min(pr.shape[1], pr_mestre.shape[1])
            pr_mestre[:, :limite] += pr[:, :limite]

    # Limita a velocidade (velocity) máxima ao padrão MIDI (127)
    pr_mestre = np.clip(pr_mestre, 0, 127)
    
    col_inicio = int(round(inicio_seg * FS))
    col_fim = int(round(fim_seg * FS))
    
    pr_cortado = pr_mestre[:, col_inicio:col_fim]
    
    # Preenchimento (padding) caso o trecho extraído seja menor que o TAMANHO_LOOP esperado
    if pr_cortado.shape[1] < TAMANHO_LOOP:
        pad_width = TAMANHO_LOOP - pr_cortado.shape[1]
        pr_cortado = np.pad(pr_cortado, ((0, 0), (0, pad_width)))
        
    return pr_cortado[:, :TAMANHO_LOOP]

def salvar_piano_roll(pr, caminho_saida):
    """
    Reconstrói um arquivo MIDI a partir de uma matriz piano-roll contínua.
    Agrupa colunas adjacentes da mesma altura musical (pitch) numa única nota contínua.
    """
    midi_out = pretty_midi.PrettyMIDI()
    inst_out = pretty_midi.Instrument(program=80)  # Program 80 = Lead 1 (square), timbre sintético
    
    for nota in range(128):
        para_tocar = np.where(pr[nota, :] > 0)[0]
        if len(para_tocar) > 0:
            inicio_atual = para_tocar[0]
            for i in range(1, len(para_tocar)):
                # Se houver quebra na continuidade (buraco), encerra a nota atual e começa outra
                if para_tocar[i] != para_tocar[i-1] + 1:
                    fim_atual = para_tocar[i-1] + 1
                    vel = int(np.mean(pr[nota, inicio_atual:fim_atual]))
                    inst_out.notes.append(pretty_midi.Note(
                        velocity=min(vel, 127), pitch=nota,
                        start=inicio_atual / FS, end=fim_atual / FS))
                    inicio_atual = para_tocar[i]
            # Fecha a última nota da sequência
            fim_atual = para_tocar[-1] + 1
            vel = int(np.mean(pr[nota, inicio_atual:fim_atual]))
            inst_out.notes.append(pretty_midi.Note(
                velocity=min(vel, 127), pitch=nota,
                start=inicio_atual / FS, end=fim_atual / FS))
                
    midi_out.instruments.append(inst_out)
    midi_out.write(caminho_saida)

def obter_notas_validas(pr_A, pr_B):
    """
    Mapeia o vocabulário tonal extraindo apenas as alturas (pitches) usadas na música A ou B.
    Isso restringe o espaço de busca e evita ruído harmônico (inspirado no MusicBlox)[cite: 4].
    """
    notas_ativas = np.where((np.sum(pr_A, axis=1) + np.sum(pr_B, axis=1)) > 0)[0]
    return notas_ativas if len(notas_ativas) > 0 else np.arange(36, 84)


# ==========================================
# CONFIGURAÇÃO DO ALGORITMO GENÉTICO (DEAP)
# ==========================================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimização de erro/perda
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def criar_individuo_caminho_A(pr_A, pr_B):
    """
    Inicializa a população partindo do fragmento 'home' (Música A),
    injetando uma pequena variação do alvo (Música B) para garantir diversidade inicial[cite: 4].
    """
    ind = pr_A.copy()
    largura = random.randint(2, 5)
    col = random.randint(0, TAMANHO_LOOP - largura)
    if random.random() < 0.3:
        ind[:, col:col+largura] = pr_B[:, col:col+largura]
    return creator.Individual(ind)

def clonar_individuo(ind):
    """Cópia profunda para garantir o funcionamento do Elitismo."""
    novo_ind = creator.Individual(ind.copy())
    novo_ind.fitness.values = ind.fitness.values
    return novo_ind

def avaliar_transicao(individuo, pr_B, mascara_invalida, w_alvo=1.0, w_tonal=2.0, w_suavidade=0.5):
    """
    Função de Fitness Multi-Critério:
    1. Distância para o alvo (Nearest Neighbour/Match)[cite: 3, 4]
    2. Penalidade Tonal (Notas fora da escala)[cite: 4]
    3. Rugosidade Horizontal (Penaliza conduções picotadas e saltos abruptos)[cite: 5]
    """
    ind_bin = (individuo > 0).astype(float)
    alvo_bin = (pr_B > 0).astype(float)
    
    # 1. Distância ponderada (70% ativação binária, 30% diferença de velocidade/dinâmica)
    erro_notas = np.mean(np.abs(ind_bin - alvo_bin))
    erro_velocidade = np.mean(np.abs(individuo - pr_B)) / 127.0
    dist_alvo = 0.7 * erro_notas + 0.3 * erro_velocidade
    
    # 2. Penaliza notas ativas que não pertencem ao vocabulário de A nem de B
    penalidade_tonal = np.mean(ind_bin[mascara_invalida, :]) if np.any(mascara_invalida) else 0.0
    
    # 3. Mede a diferença entre colunas adjacentes para favorecer notas sustentadas/fluidas[cite: 5]
    mudancas_horizontais = np.mean(np.abs(np.diff(ind_bin, axis=1)))
    
    erro_total = (w_alvo * dist_alvo) + (w_tonal * penalidade_tonal) + (w_suavidade * mudancas_horizontais)
    return (erro_total,)

def crossover_matricial(ind1, ind2):
    """Crossover de um ponto temporal: cruza as metades de dois loops."""
    ponto = random.randint(1, TAMANHO_LOOP - 1)
    temp = ind1[:, ponto:].copy()
    ind1[:, ponto:] = ind2[:, ponto:]
    ind2[:, ponto:] = temp
    return ind1, ind2

def mutacao_musical(ind, pr_B, notas_validas):
    """
    Aplica operadores de mutação estruturais baseados em processos composicionais reais[cite: 3, 4].
    Opera em blocos temporais (0.4s a 1.0s) em vez de células isoladas para manter a coesão.
    """
    tipo_op = random.choice(["add_remove_alvo", "transpor_bloco", "merge_duracao", "phase_shift"])
    largura = random.randint(4, 10)  
    col = random.randint(0, ind.shape[1] - largura)
    
    if tipo_op == "add_remove_alvo":
        # Introduz forçadamente um bloco de notas da música alvo (TraSe Add/Remove)[cite: 3]
        nota = random.choice(notas_validas)
        ind[nota, col:col+largura] = pr_B[nota, col:col+largura]
        
    elif tipo_op == "transpor_bloco":
        # Move um bloco rítmico para outra altura permitida pela escala conjunta[cite: 4]
        if len(notas_validas) >= 2:
            n1, n2 = random.sample(list(notas_validas), 2)
            ind[n2, col:col+largura] = ind[n1, col:col+largura]
            ind[n1, col:col+largura] = 0
            
    elif tipo_op == "merge_duracao":
        # Sustenta uma nota fundindo fragmentos separados (TraSe Divide/Merge)[cite: 3]
        nota = random.choice(notas_validas)
        if np.any(ind[nota, col:col+largura] > 0):
            vel_media = int(np.mean(ind[nota, col:col+largura][ind[nota, col:col+largura] > 0]))
            ind[nota, col:col+largura] = vel_media
            
    elif tipo_op == "phase_shift":
        # Desloca um bloco no tempo (TraSe Phase)[cite: 3]
        ind[:, col:col+largura] = np.roll(ind[:, col:col+largura], shift=2, axis=1)
        
    return ind,


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("A carregar ficheiros MIDI e extrair loops de 4 segundos...")
    
    # Prepara o segmento final (Source) e o inicial (Target)
    midi_temp_A = pretty_midi.PrettyMIDI(ARQUIVO_MUSICA_A)
    duracao_A = midi_temp_A.get_end_time()
    inicio_A = max(0.0, duracao_A - DURACAO_LOOP)
    fim_A = duracao_A
    
    pr_A = extrair_piano_roll_mesclado(ARQUIVO_MUSICA_A, inicio_A, fim_A)
    pr_B = extrair_piano_roll_mesclado(ARQUIVO_MUSICA_B, 0.0, DURACAO_LOOP)
    
    # Extrai o vocabulário tonal conjunto
    notas_validas = obter_notas_validas(pr_A, pr_B)
    mascara_invalida = np.ones(128, dtype=bool)
    mascara_invalida[notas_validas] = False
    
    print(f"Loop Fonte A: {inicio_A:.1f}s a {fim_A:.1f}s | Loop Alvo B: 0.0s a {DURACAO_LOOP:.1f}s")
    print(f"Notas ativas identificadas na escala conjunta: {len(notas_validas)}")
    
    # Registo de funções no DEAP
    toolbox.register("individual", criar_individuo_caminho_A, pr_A=pr_A, pr_B=pr_B)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", avaliar_transicao, pr_B=pr_B, mascara_invalida=mascara_invalida)
    toolbox.register("mate", crossover_matricial)
    toolbox.register("mutate", mutacao_musical, pr_B=pr_B, notas_validas=notas_validas)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("clone", clonar_individuo)
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Rastreador de estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg"]
    
    # Avaliação inicial (Geração 0)
    fits = toolbox.map(toolbox.evaluate, pop)
    for fit, ind in zip(fits, pop):
        ind.fitness.values = fit
        
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(logbook.stream)
    
    # Array que armazena os frames evolutivos sucessivos[cite: 3]
    frames_transicao = [pr_A.copy()]
    
    # Loop Evolutivo
    for gen in range(1, NGEN + 1):
        # Elitismo: salva o melhor da geração atual
        elite = toolbox.clone(tools.selBest(pop, k=1)[0])
        
        # Aplica Crossover e Mutação
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        
        # Reavalia apenas indivíduos modificados
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid_ind)
        for fit, ind in zip(fits, invalid_ind):
            ind.fitness.values = fit
            
        # Seleção da nova população e reinserção do elite[cite: 5]
        pop = toolbox.select(offspring, k=len(pop) - 1)
        pop.append(elite)  
        
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(logbook.stream)
        
        # Armazena o frame correspondente ao checkpoint para montagem temporal da transição
        if gen in CHECKPOINTS:
            melhor_gen = np.array(tools.selBest(pop, k=1)[0], copy=True)
            frames_transicao.append(melhor_gen)

    frames_transicao.append(pr_B.copy())
    
    # Concatenação sequencial dos frames (Trajetória da Transição)[cite: 3]
    piano_roll_final = np.hstack(frames_transicao)
    caminho_saida = "transicao_daft_punk.mid"
    salvar_piano_roll(piano_roll_final, caminho_saida)
    
    print(f"\nSucesso! Transição de {piano_roll_final.shape[1]/FS:.1f}s guardada em '{caminho_saida}'.")
    
    # ==========================================
    # GERAÇÃO DO GRÁFICO CIENTÍFICO (IEEE/ACM)
    # ==========================================
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.autolayout": True
    })

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)

    # Curvas de convergência do Algoritmo Genético[cite: 5]
    ax.plot(generations, min_fitness, label="Best Fitness (Minimum Error)", color="#1b9e77", linewidth=1.8)
    ax.plot(generations, avg_fitness, label="Mean Population Fitness", color="#386cb0", linestyle="--", linewidth=1.4)

    # Marcação visual de onde os frames foram capturados[cite: 3]
    for i, cp in enumerate(CHECKPOINTS):
        ax.axvline(
            x=cp, 
            color="#7f7f7f", 
            linestyle=":", 
            linewidth=1.0, 
            alpha=0.7, 
            label="Morphing Checkpoint" if i == 0 else ""
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value (Total Objective Loss)")
    ax.set_title("Genetic Algorithm Convergence Curve", pad=10)
    
    ax.set_xlim(0, NGEN)
    ax.grid(True, linestyle="--", alpha=0.4, color="gray")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", loc="upper right")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("ga_convergence_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig("ga_convergence_curve.pdf", bbox_inches="tight")
    print("Scientific plots saved as 'ga_convergence_curve.png' and 'ga_convergence_curve.pdf'.")
    plt.show()