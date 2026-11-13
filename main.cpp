// ============================================================================
//  Snake Game + Algoritmo Genético com controlador Perceptron (neuroevolução)
//
//  - Cada indivíduo é uma rede neural (perceptron). O genoma são os PESOS.
//  - A rede "enxerga" o jogo a cada passo (sensores) e decide: seguir reto,
//    virar à esquerda ou virar a direita.
//  - Evolução darwiniana: variação aleatória (mutação/crossover), herança
//    (filhos herdam genes dos pais), seleção natural (torneio por fitness).
//    Não há aprendizado durante a vida (sem backpropagation / Lamarck).
//  - Mutação avançada: gaussiana AUTO-ADAPTATIVA (Estratégias Evolutivas).
//    Cada indivíduo carrega no genoma sua própria "taxa de mutação" (sigma),
//    que também é herdada e mutada; a seleção favorece indiretamente os
//    sigmas que geram bons filhos.
//  - Gráfico em JANELA SEPARADA (o programa abre uma 2ª cópia de si mesmo
//    com "--grafico", que lê o evolucao.csv):
//      painel de cima : fitness do melhor e fitness médio da população
//      painel de baixo: diversidade genética e sigma de mutação do melhor
//    Exporta PNG em alta resolução.
//
//  Modos de execução:
//    main.exe              treino + cobra na tela + janela do gráfico
//    main.exe --sem-janela treino rápido só no terminal (gera o CSV)
//    main.exe --grafico    só a janela do gráfico (lê o CSV existente)
//    main.exe --jogar      usa a rede JÁ TREINADA (melhor_individuo.txt)
//                          para jogar fases novas, sem treinar de novo
//    main.exe --exportar   gera o gráfico (PNG) e o GIF a partir do que já
//                          foi salvo (útil depois de interromper um treino)
//    --continuar           junto com o treino (com ou sem janela): continua
//                          do último treino salvo em vez de começar do zero.
//                          Ex.: aumente NUM_GENERATIONS e rode
//                               main.exe --sem-janela --continuar
//
//  Arquivos gerados (na pasta do main.exe):
//    evolucao.csv           estatísticas de cada geração
//    melhor_individuo.txt   pesos do melhor indivíduo (o "cérebro" treinado)
//    populacao.txt          população inteira (para continuar o treino)
//    grafico_fitness.png    gráfico da evolução
//    melhor_jogando.gif     animação do melhor da última geração
// ============================================================================

#include <raylib.h>
#include <array>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>

using namespace std;

// ============================================================================
//  CONFIGURAÇÃO DO EXPERIMENTO  (o que mais vale a pena mudar)
// ============================================================================
// SENSORES_AVANCADOS
//   0 -> 8 sensores simples: perigo imediato e distância ao obstáculo
//        (frente/esq/dir) + direção da comida.
//   1 -> 30 sensores: visão em 8 direções (parede, corpo, comida),
//        espaço livre alcançável em cada movimento, posição do rabo
//        e tamanho da cobra.  Ajudam a não se prender no próprio corpo.
#ifndef SENSORES_AVANCADOS
#define SENSORES_AVANCADOS 1
#endif

// NEURONIOS_OCULTOS
//   0  -> perceptron de uma camada (entradas ligadas direto nas saídas)
//   >0 -> perceptron multicamadas com uma camada oculta (tanh)
#ifndef NEURONIOS_OCULTOS
#define NEURONIOS_OCULTOS 0
#endif

// ----------------------------- Tabuleiro / janela ---------------------------
const int CELL_COUNT = 30;                     // células por lado
const int CELL_SIZE  = 22;                     // pixels por celula
const int GAME_PX    = CELL_COUNT * CELL_SIZE; // área do jogo (quadrada)

// ----------------------------- Jogo / avaliação -----------------------------
const int MAX_STEPS      = 100000; // limite de passos por partida (alto: só para garantir que termina)
const int STARVE_LIMIT   = 300;  // passos sem comer -> morre de fome (evita loops infinitos)
const int GAMES_PER_EVAL = 3;    // partidas por indivíduo (fases diferentes)

// ----------------------------- Rede neural ----------------------------------
const int N_INPUTS  = SENSORES_AVANCADOS ? 30 : 8;
const int N_HIDDEN  = NEURONIOS_OCULTOS;
const int N_OUTPUTS = 3; // 0 = reto, 1 = esquerda, 2 = direita
const int GENOME_SIZE = (N_HIDDEN == 0)
    ? (N_INPUTS + 1) * N_OUTPUTS                               // +1 = bias
    : (N_INPUTS + 1) * N_HIDDEN + (N_HIDDEN + 1) * N_OUTPUTS;

// ----------------------------- Algoritmo genético ---------------------------
const int      POP_SIZE        = 500;
const int      NUM_GENERATIONS = 500;
const int      ELITE           = 2;    // melhores que sobrevivem intactos
const int      TOURNAMENT_K    = 3;    // tamanho do torneio
const double   CROSSOVER_RATE  = 0.7;
const double   SIGMA_INIT      = 0.5;  // passo de mutação inicial
const double   SIGMA_MIN       = 0.01;
const double   SIGMA_MAX       = 2.0;
const unsigned GA_SEED         = 2026; // semente do AG (troque para outro experimento)

double Now() {
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ============================================================================
//  Jogo
// ============================================================================
struct Cell { int x, y; };
bool operator==(Cell a, Cell b) { return a.x == b.x && a.y == b.y; }

// 4 direções de movimento: 0 = cima, 1 = direita, 2 = baixo, 3 = esquerda
const Cell DIRS[4] = { {0, -1}, {1, 0}, {0, 1}, {-1, 0} };
// 8 direções de visão (sentido horário a partir de cima); DIRS[d] == DIRS8[2*d]
const Cell DIRS8[8] = { {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1} };

int Manhattan(Cell a, Cell b) { return abs(a.x - b.x) + abs(a.y - b.y); }

// Por que a partida terminou
enum EndReason { JOGANDO, PAREDE, CORPO, FOME, LIMITE, ZEROU };
const char* EndReasonText(EndReason r) {
    switch (r) {
        case PAREDE: return "bateu na parede";
        case CORPO:  return "bateu no proprio corpo";
        case FOME:   return "morreu de fome (muito tempo sem comer)";
        case LIMITE: return "atingiu o limite de passos";
        case ZEROU:  return "zerou o jogo!";
        default:     return "";
    }
}

class Game {
public:
    deque<Cell> body;
    EndReason reason = JOGANDO;
    int  dir;
    Cell food;
    bool alive;
    int  steps, foods, stepsSinceFood;
    mt19937 rng;                // gerador próprio do jogo (independente do AG)
    vector<unsigned char> grid; // ocupação do tabuleiro (consulta O(1))

    // auxiliares do flood fill (busca de espaço livre)
    mutable vector<int>  visitMark;
    mutable int          visitStamp = 0;
    mutable vector<Cell> queueBuf;

    explicit Game(unsigned seed) : rng(seed) { Reset(); }

    void Reset() {
        int y = CELL_COUNT / 2;
        body = { {6, y}, {5, y}, {4, y} };
        grid.assign(CELL_COUNT * CELL_COUNT, 0);
        for (Cell c : body) grid[c.y * CELL_COUNT + c.x]++;
        visitMark.assign(CELL_COUNT * CELL_COUNT, 0);
        queueBuf.resize(CELL_COUNT * CELL_COUNT);
        dir   = 1;
        alive = true;
        reason = JOGANDO;
        steps = foods = stepsSinceFood = 0;
        PlaceFood();
    }

    static bool Inside(Cell c) { return c.x >= 0 && c.y >= 0 && c.x < CELL_COUNT && c.y < CELL_COUNT; }

    bool InBody(Cell c, bool ignoreTail) const {
        int n = grid[c.y * CELL_COUNT + c.x];
        if (ignoreTail && c == body.back()) n--;
        return n > 0;
    }

    // Parede ou corpo (o rabo sai do lugar no próximo passo, então não conta)
    bool IsBlocked(Cell c) const { return !Inside(c) || InBody(c, true); }

    void PlaceFood() {
        uniform_int_distribution<int> d(0, CELL_COUNT - 1);
        do { food = { d(rng), d(rng) }; } while (InBody(food, false));
    }

    void Step(int newDir) {
        if (!alive) return;
        if ((newDir + 2) % 4 != dir) dir = newDir; // não deixa inverter
        Cell head = { body[0].x + DIRS[dir].x, body[0].y + DIRS[dir].y };
        bool eat = (head == food);

        if (!Inside(head)) { alive = false; reason = PAREDE; return; }
        if (!eat) { grid[body.back().y * CELL_COUNT + body.back().x]--; body.pop_back(); }
        if (InBody(head, false)) { alive = false; reason = CORPO; return; }
        body.push_front(head);
        grid[head.y * CELL_COUNT + head.x]++;
        steps++;
        stepsSinceFood++;

        if (eat) {
            foods++;
            stepsSinceFood = 0;
            if ((int)body.size() == CELL_COUNT * CELL_COUNT) { alive = false; reason = ZEROU; return; }
            PlaceFood();
        }
        if (stepsSinceFood > STARVE_LIMIT) { alive = false; reason = FOME; }
        else if (steps >= MAX_STEPS) { alive = false; reason = LIMITE; }
    }

    // Quantas células livres a cobra alcança a partir de 'start' (até 'cap').
    // Se a resposta for pequena, entrar ali é entrar num beco sem saída.
    int ReachableArea(Cell start, int cap) const {
        if (IsBlocked(start)) return 0;
        visitStamp++;
        int head = 0, tail = 0, count = 0;
        queueBuf[tail++] = start;
        visitMark[start.y * CELL_COUNT + start.x] = visitStamp;
        while (head < tail && count < cap) {
            Cell c = queueBuf[head++];
            count++;
            for (int k = 0; k < 4; k++) {
                Cell n = { c.x + DIRS[k].x, c.y + DIRS[k].y };
                if (!Inside(n)) continue;
                int idx = n.y * CELL_COUNT + n.x;
                if (visitMark[idx] == visitStamp || InBody(n, true)) continue;
                visitMark[idx] = visitStamp;
                queueBuf[tail++] = n;
            }
        }
        return count;
    }

    // Projeta 'target' no referencial da cobra: (frente, direita) em [-1, 1]
    void Relative(Cell target, double& fwd, double& right) const {
        int dx = target.x - body[0].x, dy = target.y - body[0].y;
        double len = abs(dx) + abs(dy);
        if (len == 0) len = 1;
        Cell f = DIRS[dir], r = DIRS[(dir + 1) % 4];
        fwd   = (dx * f.x + dy * f.y) / len;
        right = (dx * r.x + dy * r.y) / len;
    }

    // ------------------------------------------------------------------------
    //  Sensores (sempre no referencial da cobra)
    // ------------------------------------------------------------------------
    array<double, N_INPUTS> Sense() const {
        array<double, N_INPUTS> in{};
        Cell h = body[0];

#if SENSORES_AVANCADOS
        // [0..23] Visão em 8 direções (frente, frente-dir, dir, trás-dir, trás,
        //         trás-esq, esq, frente-esq). Para cada uma:
        //         1/distância até a parede, 1/distância até o corpo (0 se não vê),
        //         1 se vê a comida nessa linha.
        for (int k = 0; k < 8; k++) {
            Cell d = DIRS8[(2 * dir + k) % 8];
            Cell c = { h.x + d.x, h.y + d.y };
            int dist = 1;
            double bodySeen = 0, foodSeen = 0;
            while (Inside(c)) {
                if (bodySeen == 0 && InBody(c, true)) bodySeen = 1.0 / dist;
                if (c == food) foodSeen = 1.0;
                c.x += d.x; c.y += d.y; dist++;
            }
            in[k * 3 + 0] = 1.0 / dist;
            in[k * 3 + 1] = bodySeen;
            in[k * 3 + 2] = foodSeen;
        }

        // [24..26] Espaço livre alcançável se for reto / esquerda / direita
        //          (1 = espaço de sobra, perto de 0 = beco sem saída)
        int cap = min(CELL_COUNT * CELL_COUNT, 2 * (int)body.size() + 20);
        int rel[3] = { dir, (dir + 3) % 4, (dir + 1) % 4 };
        for (int k = 0; k < 3; k++) {
            Cell n = { h.x + DIRS[rel[k]].x, h.y + DIRS[rel[k]].y };
            in[24 + k] = (double)ReachableArea(n, cap) / cap;
        }

        // [27..28] Onde está o rabo (frente/trás, direita/esquerda)
        Relative(body.back(), in[27], in[28]);

        // [29] Tamanho da cobra
        in[29] = body.size() / 100.0;
#else
        // [0..2] perigo imediato (frente/esq/dir)  [3..5] 1/distância do obstáculo
        int rel[3] = { dir, (dir + 3) % 4, (dir + 1) % 4 };
        for (int k = 0; k < 3; k++) {
            Cell d = DIRS[rel[k]];
            Cell c = { h.x + d.x, h.y + d.y };
            int dist = 1;
            while (!IsBlocked(c)) { dist++; c.x += d.x; c.y += d.y; }
            in[k]     = (dist == 1) ? 1.0 : 0.0;
            in[3 + k] = 1.0 / dist;
        }
        // [6..7] direção da comida (frente/trás, direita/esquerda)
        Relative(food, in[6], in[7]);
#endif
        return in;
    }
};

// ============================================================================
//  Controlador: perceptron
// ============================================================================
int ChooseDirection(const vector<double>& w, const Game& g) {
    array<double, N_INPUTS>  in  = g.Sense();
    array<double, N_OUTPUTS> out{};
    int p = 0;

    if (N_HIDDEN == 0) {
        for (int o = 0; o < N_OUTPUTS; o++) {
            double s = w[p++];                              // bias
            for (int i = 0; i < N_INPUTS; i++) s += w[p++] * in[i];
            out[o] = s;
        }
    } else {
        array<double, (N_HIDDEN > 0 ? N_HIDDEN : 1)> hid{};
        for (int j = 0; j < N_HIDDEN; j++) {
            double s = w[p++];
            for (int i = 0; i < N_INPUTS; i++) s += w[p++] * in[i];
            hid[j] = tanh(s);
        }
        for (int o = 0; o < N_OUTPUTS; o++) {
            double s = w[p++];
            for (int j = 0; j < N_HIDDEN; j++) s += w[p++] * hid[j];
            out[o] = s;
        }
    }

    // Saída vencedora (winner-takes-all)
    int best = (int)(max_element(out.begin(), out.end()) - out.begin());
    if (best == 1) return (g.dir + 3) % 4; // esquerda
    if (best == 2) return (g.dir + 1) % 4; // direita
    return g.dir;                          // reto
}

// ============================================================================
//  Indivíduo e fitness
// ============================================================================
struct Individual {
    vector<double> w;          // pesos da rede (genes)
    double sigma = SIGMA_INIT; // passo de mutação (também é um gene)
    double fitness = 0;
    double foods = 0;          // média de comidas por partida
};

// Uma partida. O indivíduo só é avaliado: nada da "experiência" vai para o genoma.
double PlayGame(const vector<double>& w, unsigned seed, int& foodsOut) {
    Game g(seed);
    double fit = 0;
    while (g.alive) {
        Cell oldFood     = g.food;
        int  before      = Manhattan(g.body[0], oldFood);
        int  foodsBefore = g.foods;

        g.Step(ChooseDirection(w, g));

        if (g.reason == PAREDE || g.reason == CORPO || g.reason == FOME) { fit -= 20; break; } // morreu
        if (g.foods > foodsBefore) fit += 100;              // comeu
        else fit += (Manhattan(g.body[0], oldFood) < before) ? 1.0 : -1.5; // aproximou / afastou
    }
    foodsOut = g.foods;
    return fit;
}

// ============================================================================
//  Algoritmo genético
// ============================================================================
struct GenStats {
    int    gen;
    double best, mean, foods, sigma, diversity;
};

class GeneticAlgorithm {
public:
    vector<Individual> pop;
    vector<unsigned>   seeds; // fases da geração atual (iguais para todos)
    int    generation = 0;
    size_t evalIndex  = 0;
    mt19937 rng;

    GeneticAlgorithm() : rng(GA_SEED) {}

    double Gauss()   { return normal_distribution<double>(0.0, 1.0)(rng); }
    double Uniform() { return uniform_real_distribution<double>(0.0, 1.0)(rng); }
    int RandInt(int a, int b) { return uniform_int_distribution<int>(a, b)(rng); }

    void Init() {
        pop.assign(POP_SIZE, Individual());
        for (auto& ind : pop) {
            ind.w.resize(GENOME_SIZE);
            for (double& x : ind.w) x = Uniform() * 2.0 - 1.0;
        }
        NewSeeds();
    }

    // O ambiente muda a cada geração: sobrevive quem joga bem em qualquer fase,
    // e não quem decorou um caminho.
    void NewSeeds() {
        seeds.clear();
        for (int k = 0; k < GAMES_PER_EVAL; k++) seeds.push_back(rng());
        evalIndex = 0;
    }

    void Evaluate(Individual& ind) {
        double total = 0, foods = 0;
        for (unsigned s : seeds) {
            int f = 0;
            total += PlayGame(ind.w, s, f);
            foods += f;
        }
        ind.fitness = total / seeds.size();
        ind.foods   = foods / seeds.size();
    }

    // Avalia indivíduos até gastar 'budget' segundos (mantém a janela fluida).
    // Retorna true quando a geração inteira foi avaliada.
    bool EvaluateSome(double budget) {
        double start = Now();
        while (evalIndex < pop.size()) {
            Evaluate(pop[evalIndex++]);
            if (Now() - start > budget) break;
        }
        if (evalIndex < pop.size()) return false;
        sort(pop.begin(), pop.end(),
             [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        return true;
    }

    // Estatísticas da geração recém-avaliada
    GenStats Stats() const {
        GenStats s;
        s.gen   = generation + 1;
        s.best  = pop[0].fitness;
        s.foods = pop[0].foods;
        s.sigma = pop[0].sigma;

        double sum = 0;
        for (auto& ind : pop) sum += ind.fitness;
        s.mean = sum / pop.size();

        // Diversidade genética: desvio padrão de cada gene na população,
        // em média sobre todos os genes. Perto de 0 = todos iguais.
        double div = 0;
        for (int i = 0; i < GENOME_SIZE; i++) {
            double m = 0, m2 = 0;
            for (auto& ind : pop) { m += ind.w[i]; m2 += ind.w[i] * ind.w[i]; }
            m /= pop.size(); m2 /= pop.size();
            div += sqrt(max(0.0, m2 - m * m));
        }
        s.diversity = div / GENOME_SIZE;
        return s;
    }

    // Seleção natural: torneio entre K indivíduos sorteados
    const Individual& Tournament() {
        int best = RandInt(0, POP_SIZE - 1);
        for (int k = 1; k < TOURNAMENT_K; k++) {
            int c = RandInt(0, POP_SIZE - 1);
            if (pop[c].fitness > pop[best].fitness) best = c;
        }
        return pop[best];
    }

    // Mutação gaussiana auto-adaptativa (log-normal, Schwefel):
    //   sigma' = sigma * exp(tau * N(0,1))
    //   w_i'   = w_i + sigma' * N(0,1)
    // A mutação é cega: não usa o fitness nem "sabe" para onde melhorar.
    void Mutate(Individual& c) {
        const double tau = 1.0 / sqrt((double)GENOME_SIZE);
        c.sigma *= exp(tau * Gauss());
        c.sigma  = clamp(c.sigma, SIGMA_MIN, SIGMA_MAX);
        for (double& x : c.w) x += c.sigma * Gauss();
    }

    void NextGeneration() {
        vector<Individual> next;
        next.reserve(POP_SIZE);

        // Elitismo: os melhores sobrevivem (como um organismo que vive mais
        // de uma estação). Eles serão reavaliados nas fases novas.
        for (int e = 0; e < ELITE; e++) next.push_back(pop[e]);

        while ((int)next.size() < POP_SIZE) {
            const Individual& p1 = Tournament();
            const Individual& p2 = Tournament();
            Individual child = p1;

            if (Uniform() < CROSSOVER_RATE) {
                // Crossover uniforme: cada gene vem de um dos pais (50%)
                for (int i = 0; i < GENOME_SIZE; i++)
                    if (Uniform() < 0.5) child.w[i] = p2.w[i];
                child.sigma = sqrt(p1.sigma * p2.sigma); // herança do sigma
            }
            Mutate(child);
            next.push_back(child);
        }

        pop = move(next);
        generation++;
        NewSeeds();
    }
};

// ============================================================================
//  Arquivos compartilhados entre as janelas
// ============================================================================
string AppDir() { return string(GetApplicationDirectory()); }

// Cada configuração da rede tem sua própria pasta de resultados, para que
// testes diferentes (perceptron x multicamadas) nunca se sobrescrevam:
//   resultados/perceptron/
//   resultados/multicamada_16_ocultos/
//   resultados/perceptron_sensores_simples/   ...
string ConfigName() {
    string n = (N_HIDDEN == 0) ? "perceptron" : "multicamada_" + to_string(N_HIDDEN) + "_ocultos";
    if (!SENSORES_AVANCADOS) n += "_sensores_simples";
    return n;
}

// Descrição da configuração, mostrada no gráfico
string ConfigLabel() {
    string n = (N_HIDDEN == 0) ? "Perceptron de uma camada"
                               : "Perceptron multicamadas (" + to_string(N_HIDDEN) + " neurônios ocultos)";
    n += SENSORES_AVANCADOS ? " · 30 sensores" : " · 8 sensores";
    return n;
}

string ResultDir() {
    static const string dir = [] {
        string d = AppDir() + "resultados/" + ConfigName() + "/";
        error_code ec;
        filesystem::create_directories(d, ec);
        return d;
    }();
    return dir;
}

string CsvPath() { return ResultDir() + "evolucao.csv"; }
string PngPath() { return ResultDir() + "grafico_fitness.png"; }

const char* CSV_HEADER = "geracao,melhor_fitness,fitness_medio,comidas_melhor,sigma_melhor,diversidade\n";

void WriteCsvLine(ofstream& csv, const GenStats& s) {
    csv << s.gen << "," << s.best << "," << s.mean << "," << s.foods << ","
        << s.sigma << "," << s.diversity << "\n";
    csv.flush();
}

void PrintStats(const GenStats& s) {
    printf("Geracao %3d | melhor %8.1f | media %8.1f | comidas %5.1f | sigma %.3f | diversidade %.3f\n",
           s.gen, s.best, s.mean, s.foods, s.sigma, s.diversity);
    fflush(stdout);
}

vector<GenStats> ReadCsv(const string& path) {
    vector<GenStats> d;
    ifstream f(path);
    string line;
    getline(f, line); // cabeçalho
    while (getline(f, line)) {
        GenStats s;
        if (sscanf(line.c_str(), "%d,%lf,%lf,%lf,%lf,%lf",
                   &s.gen, &s.best, &s.mean, &s.foods, &s.sigma, &s.diversity) == 6)
            d.push_back(s);
    }
    return d;
}

// Abre uma segunda cópia deste programa no modo "--grafico" (janela separada)
void LaunchChartWindow(const char* argv0) {
    string exe = AppDir() + GetFileName(argv0);
#ifdef _WIN32
    if (exe.size() < 4 || exe.substr(exe.size() - 4) != ".exe") exe += ".exe";
    string cmd = "start \"\" \"" + exe + "\" --grafico";
#else
    string cmd = "\"" + exe + "\" --grafico &";
#endif
    if (system(cmd.c_str()) != 0) cerr << "Nao foi possivel abrir a janela do grafico" << endl;
}

// ============================================================================
//  Salvar e carregar o "cérebro" treinado
//  O resultado do treino é só uma lista de números (os pesos da rede).
//  Guardando esses números num arquivo, dá para usar a rede depois sem
//  treinar de novo (modo --jogar).
// ============================================================================
string BestPath() { return ResultDir() + "melhor_individuo.txt"; }
string GifPath()  { return ResultDir() + "melhor_jogando.gif"; }

void SaveBest(const Individual& ind, int gen) {
    string tmp = BestPath() + ".tmp";
    {
        ofstream f(tmp);
        f << "# Snake evolutivo - pesos do melhor individuo (o cerebro treinado)\n";
        f << "sensores_avancados " << SENSORES_AVANCADOS << "\n";
        f << "neuronios_ocultos " << N_HIDDEN << "\n";
        f << "geracao " << gen << "\n";
        f << "fitness " << ind.fitness << "\n";
        f << "comidas " << ind.foods << "\n";
        f << "genes " << ind.w.size() << "\n";
        f.precision(17);
        for (double x : ind.w) f << x << "\n";
    }
    // troca só depois de gravar tudo (seguro se o programa for interrompido)
    remove(BestPath().c_str());
    rename(tmp.c_str(), BestPath().c_str());
}

bool LoadBest(const string& path, Individual& ind, int& gen, string& err) {
    ifstream f(path);
    if (!f) {
        err = "Arquivo nao encontrado:\n" + path + "\n\nRode o treino primeiro.";
        return false;
    }
    string line, key;
    getline(f, line); // comentário
    int sens = -1, hid = -1, n = -1;
    f >> key >> sens >> key >> hid >> key >> gen >> key >> ind.fitness >> key >> ind.foods >> key >> n;
    if (!f) { err = "Arquivo invalido: " + path; return false; }
    if (sens != SENSORES_AVANCADOS || hid != N_HIDDEN || n != GENOME_SIZE) {
        err = "A rede salva nao combina com a configuracao atual.\n"
              "Arquivo: SENSORES_AVANCADOS=" + to_string(sens) + "  NEURONIOS_OCULTOS=" + to_string(hid) +
              "\nCodigo:  SENSORES_AVANCADOS=" + to_string(SENSORES_AVANCADOS) +
              "  NEURONIOS_OCULTOS=" + to_string(N_HIDDEN) +
              "\n\nAjuste as constantes no main.cpp ou treine de novo.";
        return false;
    }
    ind.w.resize(n);
    for (double& x : ind.w) f >> x;
    if (!f) { err = "Arquivo incompleto: " + path; return false; }
    return true;
}

// ============================================================================
//  GIF animado, escrito direto em arquivo (não precisa de janela)
//  Cada quadro guarda só o retângulo que mudou; o resto fica transparente,
//  o que deixa o arquivo pequeno.
// ============================================================================
const int GIF_CELL             = 12; // pixels por célula no GIF
const int GIF_PASSOS_POR_QUADRO = 3;  // passos do jogo por quadro (acelera o GIF)
const int GIF_ATRASO_CS        = 3;  // atraso entre quadros, em centésimos de segundo
const int GIF_MAX_QUADROS       = 2000; // partidas longas pulam mais passos por quadro (GIF de ~1 min)

class GifWriter {
public:
    static const int PAL    = 8; // cores na paleta
    static const int TRANSP = 7; // índice usado como transparente

    bool Open(const string& path, int w, int h) {
        f = fopen(path.c_str(), "wb");
        if (!f) return false;
        W = w; H = h; first = true;
        fwrite("GIF89a", 1, 6, f);
        Put16(W); Put16(H);
        fputc(0x80 | (2 << 4) | 2, f); // paleta global com 8 cores
        fputc(0, f); fputc(0, f);
        const uint8_t pal[PAL][3] = {
            { 0, 0, 0 },     { 230, 41, 55 }, { 0, 121, 241 }, { 0, 228, 48 },
            { 40, 40, 40 },  { 255, 255, 255 }, { 253, 249, 0 }, { 255, 0, 255 },
        };
        fwrite(pal, 1, sizeof(pal), f);
        // Extensão NETSCAPE: repetir o GIF para sempre
        const uint8_t loop[] = { 0x21, 0xFF, 0x0B, 'N','E','T','S','C','A','P','E','2','.','0', 0x03, 0x01, 0, 0, 0 };
        fwrite(loop, 1, sizeof(loop), f);
        return true;
    }

    void Frame(const vector<uint8_t>& px, int delayCs) {
        int x0 = 0, y0 = 0, x1 = W - 1, y1 = H - 1;
        bool transparent = !first;
        if (!first) {
            x0 = W; y0 = H; x1 = -1; y1 = -1;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (px[y * W + x] != prev[y * W + x]) {
                        x0 = min(x0, x); x1 = max(x1, x);
                        y0 = min(y0, y); y1 = max(y1, y);
                    }
            if (x1 < 0) { x0 = y0 = x1 = y1 = 0; } // nada mudou: quadro de 1 pixel
        }
        int w = x1 - x0 + 1, h = y1 - y0 + 1;
        vector<uint8_t> sub(w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int idx = (y0 + y) * W + (x0 + x);
                sub[y * w + x] = (transparent && px[idx] == prev[idx]) ? TRANSP : px[idx];
            }

        // Graphic Control Extension (atraso + transparência)
        fputc(0x21, f); fputc(0xF9, f); fputc(4, f);
        fputc((1 << 2) | (transparent ? 1 : 0), f);
        Put16(delayCs);
        fputc(transparent ? TRANSP : 0, f);
        fputc(0, f);
        // Image Descriptor
        fputc(0x2C, f); Put16(x0); Put16(y0); Put16(w); Put16(h); fputc(0, f);
        WriteLzw(sub);

        prev = px;
        first = false;
    }

    void Close() {
        if (!f) return;
        fputc(0x3B, f);
        fclose(f);
        f = nullptr;
    }

private:
    FILE* f = nullptr;
    int W = 0, H = 0;
    bool first = true;
    vector<uint8_t> prev;

    void Put16(int v) { fputc(v & 0xFF, f); fputc((v >> 8) & 0xFF, f); }

    // Compressão LZW (formato exigido pelo GIF)
    void WriteLzw(const vector<uint8_t>& data) {
        const int minCode = 3, clearCode = 1 << minCode, eoi = clearCode + 1;
        fputc(minCode, f);
        vector<uint16_t> dict(4096 * PAL, 0);
        vector<uint8_t> bytes;
        uint32_t acc = 0;
        int nbits = 0, codeSize = minCode + 1, maxCode = eoi;
        auto put = [&](int code) {
            acc |= (uint32_t)code << nbits;
            nbits += codeSize;
            while (nbits >= 8) { bytes.push_back(acc & 0xFF); acc >>= 8; nbits -= 8; }
        };

        put(clearCode);
        int cur = data[0];
        for (size_t i = 1; i < data.size(); i++) {
            int v = data[i];
            uint16_t& next = dict[cur * PAL + v];
            if (next) { cur = next; continue; }
            put(cur);
            next = ++maxCode;
            if (maxCode >= (1 << codeSize)) codeSize++;
            if (maxCode == 4095) {
                put(clearCode);
                fill(dict.begin(), dict.end(), 0);
                codeSize = minCode + 1;
                maxCode = eoi;
            }
            cur = v;
        }
        put(cur);
        put(eoi);
        if (nbits > 0) bytes.push_back(acc & 0xFF);

        for (size_t i = 0; i < bytes.size(); i += 255) {
            size_t len = min<size_t>(255, bytes.size() - i);
            fputc((int)len, f);
            fwrite(&bytes[i], 1, len, f);
        }
        fputc(0, f);
    }
};

// Desenha o tabuleiro numa imagem de índices de cor (mesmas cores da janela)
void RenderGameIndices(const Game& g, vector<uint8_t>& px) {
    const int W = CELL_COUNT * GIF_CELL;
    fill(px.begin(), px.end(), 0);
    auto rect = [&](Cell c, int inset, uint8_t color) {
        for (int y = c.y * GIF_CELL + inset; y < (c.y + 1) * GIF_CELL - inset; y++)
            for (int x = c.x * GIF_CELL + inset; x < (c.x + 1) * GIF_CELL - inset; x++)
                px[y * W + x] = color;
    };
    rect(g.food, 0, 1);
    for (size_t i = 0; i < g.body.size(); i++) rect(g.body[i], 1, i == 0 ? 2 : 3);
}

// Grava uma partida inteira do indivíduo como GIF. Retorna a partida jogada.
Game RecordGif(const Individual& ind, unsigned seed, const string& path) {
    const int W = CELL_COUNT * GIF_CELL;

    // Joga uma vez antes só para saber o tamanho da partida
    // (o jogo é determinístico: a mesma semente gera a mesma partida)
    Game probe(seed);
    while (probe.alive) probe.Step(ChooseDirection(ind.w, probe));
    int stepsPerFrame = max(GIF_PASSOS_POR_QUADRO, (probe.steps + GIF_MAX_QUADROS - 1) / GIF_MAX_QUADROS);

    Game g(seed);
    GifWriter gif;
    if (!gif.Open(path, W, W)) { cerr << "Nao foi possivel criar " << path << endl; return g; }
    vector<uint8_t> px(W * W);
    RenderGameIndices(g, px);
    gif.Frame(px, GIF_ATRASO_CS);
    while (g.alive) {
        for (int k = 0; k < stepsPerFrame && g.alive; k++)
            g.Step(ChooseDirection(ind.w, g));
        RenderGameIndices(g, px);
        gif.Frame(px, GIF_ATRASO_CS);
    }
    gif.Frame(px, 300); // segura a última imagem por 3 s antes de repetir
    gif.Close();
    return g;
}
// ============================================================================
//  Continuar o treino: salva a população INTEIRA a cada geração
//  (pesos, sigma e fitness de todos, mais o estado do sorteio aleatório).
//  Continuar a partir daqui dá exatamente o mesmo resultado que um treino
//  que nunca tivesse parado.
// ============================================================================
string CheckpointPath() { return ResultDir() + "populacao.txt"; }

// Versões antigas do programa salvavam tudo direto na pasta do main.exe.
// Se houver resultados antigos com a MESMA configuração e a pasta nova ainda
// estiver vazia, eles são movidos para a pasta da configuração.
void MigrateOldResults() {
    string oldBest = AppDir() + "melhor_individuo.txt";
    error_code ec;
    if (!filesystem::exists(oldBest, ec) || filesystem::exists(CsvPath(), ec)) return;

    ifstream f(oldBest);
    string line, key;
    int sens = -1, hid = -1;
    getline(f, line);
    f >> key >> sens >> key >> hid;
    f.close();
    if (sens != SENSORES_AVANCADOS || hid != N_HIDDEN) return; // é de outra configuração

    const char* files[] = { "evolucao.csv", "melhor_individuo.txt", "populacao.txt",
                            "grafico_fitness.png", "melhor_jogando.gif" };
    for (const char* name : files) {
        string from = AppDir() + name;
        if (filesystem::exists(from, ec)) filesystem::rename(from, ResultDir() + name, ec);
    }
    printf("Resultados antigos movidos para: %s\n", ResultDir().c_str());
}

void SaveCheckpoint(const GeneticAlgorithm& ga) {
    string tmp = CheckpointPath() + ".tmp";
    {
        ofstream f(tmp);
        f << "# Snake evolutivo - populacao completa (para continuar o treino)\n";
        f << "sensores_avancados " << SENSORES_AVANCADOS << "\n"
          << "neuronios_ocultos " << N_HIDDEN << "\n"
          << "genes " << GENOME_SIZE << "\n"
          << "populacao " << ga.pop.size() << "\n"
          << "geracao " << ga.generation << "\n";
        f << "rng " << ga.rng << "\n";
        f << "fases " << ga.seeds.size();
        for (unsigned sd : ga.seeds) f << " " << sd;
        f << "\n";
        f.precision(17);
        for (auto& ind : ga.pop) {
            f << ind.sigma << " " << ind.fitness << " " << ind.foods;
            for (double x : ind.w) f << " " << x;
            f << "\n";
        }
    }
    // Grava num arquivo temporário e só depois troca: se o programa for
    // interrompido no meio da gravação, o checkpoint anterior não se perde.
    remove(CheckpointPath().c_str());
    rename(tmp.c_str(), CheckpointPath().c_str());
}

bool LoadCheckpoint(GeneticAlgorithm& ga, string& err) {
    ifstream f(CheckpointPath());
    if (!f) { err = "Nao encontrei " + CheckpointPath() + "\nRode um treino antes de usar --continuar."; return false; }
    string line, key;
    getline(f, line); // comentário
    int sens = -1, hid = -1, genes = -1, popn = -1, gen = -1;
    f >> key >> sens >> key >> hid >> key >> genes >> key >> popn >> key >> gen;
    if (!f) { err = "Arquivo invalido: " + CheckpointPath(); return false; }
    if (sens != SENSORES_AVANCADOS || hid != N_HIDDEN || genes != GENOME_SIZE || popn != POP_SIZE) {
        err = "O treino salvo nao combina com a configuracao atual\n"
              "(SENSORES_AVANCADOS, NEURONIOS_OCULTOS ou POP_SIZE mudaram).\n"
              "Volte as constantes ou comece um treino novo (sem --continuar).";
        return false;
    }
    f >> key >> ga.rng;
    size_t ns = 0;
    f >> key >> ns;
    ga.seeds.resize(ns);
    for (unsigned& sd : ga.seeds) f >> sd;
    ga.pop.assign(popn, Individual());
    for (auto& ind : ga.pop) {
        f >> ind.sigma >> ind.fitness >> ind.foods;
        ind.w.resize(genes);
        for (double& x : ind.w) f >> x;
    }
    if (!f) { err = "Arquivo incompleto: " + CheckpointPath(); return false; }
    ga.generation = gen;
    ga.evalIndex  = ga.pop.size();
    return true;
}

// Prepara o algoritmo: do zero, ou continuando do último checkpoint.
// No segundo caso, mantém o histórico do CSV até a geração salva.
bool StartTraining(GeneticAlgorithm& ga, bool cont, ofstream& csv, string& err) {
    if (!cont) {
        ga.Init();
        csv.open(CsvPath());
        csv << CSV_HEADER;
        csv.flush();
        return true;
    }
    if (!LoadCheckpoint(ga, err)) return false;
    int done = ga.generation + 1;
    if (done >= NUM_GENERATIONS) {
        err = "O treino salvo ja tem " + to_string(done) + " geracoes.\n"
              "Aumente NUM_GENERATIONS no main.cpp para continuar.";
        return false;
    }
    vector<GenStats> old = ReadCsv(CsvPath());
    csv.open(CsvPath());
    csv << CSV_HEADER;
    for (auto& st : old)
        if (st.gen <= done) WriteCsvLine(csv, st);
    printf("Continuando o treino da geracao %d ate a %d\n", done, NUM_GENERATIONS);
    ga.NextGeneration();
    return true;
}

// ============================================================================
//  Janela 1: treino + melhor indivíduo jogando
// ============================================================================
void DrawGame(const Game& g) {
    DrawRectangle(0, 0, GAME_PX, GAME_PX, BLACK);
    DrawRectangle(g.food.x * CELL_SIZE, g.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE, RED);
    for (size_t i = 0; i < g.body.size(); i++)
        DrawRectangle(g.body[i].x * CELL_SIZE + 1, g.body[i].y * CELL_SIZE + 1,
                      CELL_SIZE - 2, CELL_SIZE - 2, i == 0 ? BLUE : GREEN);
}

int RunTraining(const char* argv0, bool cont) {
    GeneticAlgorithm ga;
    ofstream csv;
    string err;
    if (!StartTraining(ga, cont, csv, err)) { cerr << err << endl; return 1; }

    LaunchChartWindow(argv0);

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(GAME_PX, GAME_PX, "Snake - melhor individuo da geracao");
    SetWindowPosition(30, 60);
    SetTargetFPS(60);

    int  generationsDone = ga.generation; // > 0 quando continua um treino
    bool trainingDone    = false;

    // Melhor da geração mais recente (entra na tela quando a partida atual acabar)
    Individual latestBest;
    unsigned   latestSeed = 0;
    int        latestGen  = -1;

    Individual shown;
    int  shownGen   = -1;
    Game replay(0);
    bool hasReplay  = false;
    double lastStep = 0, stepInterval = 0.04;
    double endTime  = -1; // quando a partida exibida terminou

    while (!WindowShouldClose()) {
        // ---------------- Treino (um pedaco por frame) ----------------
        if (!trainingDone && ga.EvaluateSome(0.012)) {
            GenStats s = ga.Stats();
            WriteCsvLine(csv, s);
            PrintStats(s);

            latestBest = ga.pop[0];
            latestSeed = ga.seeds[0];
            latestGen  = ga.generation;
            generationsDone++;
            SaveBest(latestBest, ga.generation + 1);
            SaveCheckpoint(ga);

            if (ga.generation + 1 >= NUM_GENERATIONS) {
                trainingDone = true;
                RecordGif(latestBest, latestSeed, GifPath());
            }
            else ga.NextGeneration();
        }

        // ---------------- Exibição do melhor indivíduo ----------------
        if (IsKeyPressed(KEY_UP))   stepInterval = max(0.005, stepInterval / 1.5);
        if (IsKeyPressed(KEY_DOWN)) stepInterval = min(0.3,   stepInterval * 1.5);

        if (!hasReplay && latestGen >= 0) {
            shown = latestBest; shownGen = latestGen;
            replay = Game(latestSeed); hasReplay = true;
        }
        if (hasReplay && GetTime() - lastStep >= stepInterval) {
            lastStep = GetTime();
            if (replay.alive) {
                replay.Step(ChooseDirection(shown.w, replay));
                if (!replay.alive) endTime = GetTime();
            } else if (GetTime() - endTime > 2.0) {
                // Mostra o motivo por 2 s e passa para o melhor da geração mais recente
                shown = latestBest; shownGen = latestGen;
                replay = Game(latestSeed);
            }
        }

        // ---------------- Desenho ----------------
        BeginDrawing();
        ClearBackground(BLACK);
        if (hasReplay) {
            DrawGame(replay);
            DrawText(TextFormat("Exibindo: melhor da geracao %d", shownGen + 1), 8, 8, 18, WHITE);
            DrawText(TextFormat("Comidas: %d   Passos: %d", replay.foods, replay.steps), 8, 30, 18, WHITE);
            if (!replay.alive) {
                const char* t = TextFormat("Fim: %s", EndReasonText(replay.reason));
                int tw = MeasureText(t, 22);
                DrawRectangle(GAME_PX / 2 - tw / 2 - 12, GAME_PX / 2 - 22, tw + 24, 44, Fade(BLACK, 0.8f));
                DrawText(t, GAME_PX / 2 - tw / 2, GAME_PX / 2 - 11, 22, YELLOW);
            }
        } else {
            DrawText("Avaliando a primeira geracao...", GAME_PX / 2 - 150, GAME_PX / 2, 20, WHITE);
        }
        DrawText(TextFormat("Treino: %d / %d %s", generationsDone, NUM_GENERATIONS,
                            trainingDone ? "(concluido)" : ""), 8, GAME_PX - 46, 18, LIGHTGRAY);
        DrawText("Setas cima/baixo: velocidade", 8, GAME_PX - 24, 16, GRAY);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

// ============================================================================
//  Janela 2: gráfico (lê o CSV escrito pela janela de treino)
// ============================================================================
const Color C_TEXT = { 30, 30, 40, 255 };
const Color C_GRID = { 225, 228, 235, 255 };

struct Series {
    const char* name;
    Color color;
    vector<double> y;
};

// Passo "redondo" para os eixos (1, 2, 5 x 10^n)
double NiceStep(double range, int targetTicks) {
    double raw  = range / targetTicks;
    double mag  = pow(10.0, floor(log10(raw)));
    double norm = raw / mag;
    double step = (norm < 1.5) ? 1 : (norm < 3.5) ? 2 : (norm < 7.5) ? 5 : 10;
    return step * mag;
}

// Um painel com eixos, grade, várias curvas e legenda
void DrawPanel(Rectangle r, const vector<int>& gens, const vector<Series>& series,
               const char* yLabel, bool xLabels, const Font& font, float u, int yTicks) {
    // O eixo vai até NUM_GENERATIONS, ou mais se o CSV tiver mais gerações
    // (por exemplo, um treino continuado além do valor deste executável)
    double xMax = max(NUM_GENERATIONS, gens.empty() ? 0 : gens.back());
    double yLo = 0, yHi = 1;
    bool any = false;
    for (auto& s : series)
        for (double v : s.y) {
            if (!any) { yLo = min(0.0, v); yHi = v; any = true; }
            yLo = min(yLo, v); yHi = max(yHi, v);
        }
    if (yHi - yLo <= 1e-9) yHi = yLo + 1;
    double yStep = NiceStep(yHi - yLo, yTicks);
    yLo = floor(yLo / yStep) * yStep;
    yHi = ceil(yHi / yStep) * yStep;
    double xStep = NiceStep(xMax, 8);

    auto PX = [&](double g) { return r.x + (float)(g / xMax) * r.width; };
    auto PY = [&](double v) { return r.y + r.height - (float)((v - yLo) / (yHi - yLo)) * r.height; };

    float fs = 20 * u;
    float lw = max(1.0f, u);
    bool decimals = yStep < 1;

    for (double v = yLo; v <= yHi + yStep * 0.5; v += yStep) {
        float y = PY(v);
        DrawLineEx({ r.x, y }, { r.x + r.width, y }, lw, C_GRID);
        const char* t = decimals ? TextFormat("%.2f", v) : TextFormat("%.0f", v);
        Vector2 m = MeasureTextEx(font, t, fs, 0);
        DrawTextEx(font, t, { r.x - m.x - 12 * u, y - m.y / 2 }, fs, 0, C_TEXT);
    }
    for (double g = 0; g <= xMax + 0.5; g += xStep) {
        float x = PX(g);
        DrawLineEx({ x, r.y }, { x, r.y + r.height }, lw, C_GRID);
        if (xLabels) {
            const char* t = TextFormat("%.0f", g);
            Vector2 m = MeasureTextEx(font, t, fs, 0);
            DrawTextEx(font, t, { x - m.x / 2, r.y + r.height + 10 * u }, fs, 0, C_TEXT);
        }
    }
    DrawRectangleLinesEx(r, 1.5f * u, C_TEXT);

    // Rótulo do eixo Y (vertical)
    float ls = 23 * u;
    Vector2 ym = MeasureTextEx(font, yLabel, ls, 0);
    DrawTextPro(font, yLabel, { r.x - 105 * u, r.y + (r.height + ym.x) / 2 }, { 0, 0 }, -90, ls, 0, C_TEXT);

    // Curvas
    float thick = 2.8f * u;
    for (auto& s : series) {
        for (size_t i = 1; i < s.y.size(); i++)
            DrawLineEx({ PX(gens[i - 1]), PY(s.y[i - 1]) }, { PX(gens[i]), PY(s.y[i]) }, thick, s.color);
        if (!s.y.empty()) DrawCircleV({ PX(gens.back()), PY(s.y.back()) }, thick * 1.4f, s.color);
    }

    // Legenda (canto superior esquerdo do painel)
    float lx = r.x + 16 * u, ly = r.y + 14 * u, lsz = 19 * u;
    float wMax = 0;
    for (auto& s : series) wMax = max(wMax, MeasureTextEx(font, s.name, lsz, 0).x);
    Rectangle box = { lx - 8 * u, ly - 6 * u, wMax + 60 * u, series.size() * 28 * u + 8 * u };
    DrawRectangleRec(box, Fade(WHITE, 0.92f));
    DrawRectangleLinesEx(box, u, C_GRID);
    for (auto& s : series) {
        DrawLineEx({ lx, ly + 11 * u }, { lx + 30 * u, ly + 11 * u }, thick * 1.3f, s.color);
        DrawTextEx(font, s.name, { lx + 40 * u, ly }, lsz, 0, C_TEXT);
        ly += 28 * u;
    }
}

// Figura completa em W x H (serve para a tela e para o PNG)
void DrawChart(const vector<GenStats>& d, int W, int H, const Font& font) {
    const float u = H / 900.0f; // tudo escala junto com a figura
    ClearBackground(WHITE);

    const char* title = "Evolução da população";
    float ts = 34 * u;
    Vector2 tm = MeasureTextEx(font, title, ts, 0);
    DrawTextEx(font, title, { (W - tm.x) / 2, 18 * u }, ts, 0, C_TEXT);

    string sub = ConfigLabel();
    float ss = 20 * u;
    Vector2 sm = MeasureTextEx(font, sub.c_str(), ss, 0);
    DrawTextEx(font, sub.c_str(), { (W - sm.x) / 2, 60 * u }, ss, 0, GRAY);

    vector<int> gens;
    Series best{ "Melhor indivíduo",       { 31, 119, 180, 255 }, {} };
    Series mean{ "Média da população",     { 255, 127, 14, 255 }, {} };
    Series div { "Diversidade genética",   { 44, 160, 44, 255 },  {} };
    Series sig { "Sigma de mutação (melhor)", { 148, 103, 189, 255 }, {} };
    for (auto& s : d) {
        gens.push_back(s.gen);
        best.y.push_back(s.best);
        mean.y.push_back(s.mean);
        div.y.push_back(s.diversity);
        sig.y.push_back(s.sigma);
    }

    float left = 150 * u, right = 50 * u, top = 105 * u, bottom = 110 * u, gap = 40 * u;
    float avail = H - top - bottom - gap;
    Rectangle pTop = { left, top, W - left - right, avail * 0.62f };
    Rectangle pBot = { left, top + pTop.height + gap, W - left - right, avail * 0.38f };

    DrawPanel(pTop, gens, { best, mean }, "Fitness", false, font, u, 6);
    DrawPanel(pBot, gens, { div, sig }, "Desvio (pesos)", true, font, u, 3);

    float ls = 23 * u;
    Vector2 xm = MeasureTextEx(font, "Geração", ls, 0);
    DrawTextEx(font, "Geração", { pBot.x + (pBot.width - xm.x) / 2, H - 62 * u }, ls, 0, C_TEXT);

    if (d.empty()) {
        const char* t = "Aguardando a primeira geração...";
        Vector2 m = MeasureTextEx(font, t, 22 * u, 0);
        DrawTextEx(font, t, { pTop.x + (pTop.width - m.x) / 2, pTop.y + pTop.height / 2 }, 22 * u, 0, GRAY);
        return;
    }

    // Valores da última geração (canto inferior direito do painel de cima)
    const GenStats& l = d.back();
    const char* info = TextFormat("Geração %d/%d   melhor %.0f   média %.0f   comidas %.1f",
                                  l.gen, max(NUM_GENERATIONS, l.gen), l.best, l.mean, l.foods);
    float is = 19 * u;
    Vector2 im = MeasureTextEx(font, info, is, 0);
    Rectangle box = { pTop.x + pTop.width - im.x - 32 * u, pTop.y + pTop.height - im.y - 26 * u,
                      im.x + 20 * u, im.y + 12 * u };
    DrawRectangleRec(box, Fade(WHITE, 0.92f));
    DrawRectangleLinesEx(box, u, C_GRID);
    DrawTextEx(font, info, { box.x + 10 * u, box.y + 6 * u }, is, 0, C_TEXT);
}

// Renderiza em 2x e reduz (supersampling = linhas suaves) e salva o PNG
void ExportChartPNG(const vector<GenStats>& d, const Font& font, const string& path) {
    const int W = 2400, H = 1500, SS = 2;
    RenderTexture2D rt = LoadRenderTexture(W * SS, H * SS);
    BeginTextureMode(rt);
    DrawChart(d, W * SS, H * SS, font);
    EndTextureMode();
    Image img = LoadImageFromTexture(rt.texture);
    ImageFlipVertical(&img);
    ImageResize(&img, W, H);
    ExportImage(img, path.c_str());
    UnloadImage(img);
    UnloadRenderTexture(rt);
}

Font LoadChartFont() {
    // Codepoints ASCII + Latin-1 (acentos do português)
    vector<int> cps;
    for (int c = 32; c <= 255; c++) cps.push_back(c);
    const char* candidates[] = {
        "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    };
    for (const char* path : candidates) {
        if (FileExists(path)) {
            Font f = LoadFontEx(path, 128, cps.data(), (int)cps.size());
            if (f.texture.id != 0) {
                GenTextureMipmaps(&f.texture);
                SetTextureFilter(f.texture, TEXTURE_FILTER_TRILINEAR);
                return f;
            }
        }
    }
    return GetFontDefault();
}

int RunChartWindow() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1100, 760, "Evolucao da populacao");
    SetWindowPosition(30 + GAME_PX + 20, 40);
    SetWindowMinSize(500, 350);
    SetTargetFPS(30);

    Font font = LoadChartFont();
    vector<GenStats> data;
    double lastRead = -10;
    bool exported = false;
    string msg;
    double msgTime = -10;

    while (!WindowShouldClose()) {
        if (GetTime() - lastRead > 0.5) {
            data = ReadCsv(CsvPath());
            lastRead = GetTime();
            if (data.size() < (size_t)NUM_GENERATIONS) exported = false; // treino reiniciado
        }

        bool finished = !data.empty() && data.back().gen >= NUM_GENERATIONS;
        if ((finished && !exported) || IsKeyPressed(KEY_S)) {
            ExportChartPNG(data, font, PngPath());
            exported = true;
            msg = "PNG salvo: " + PngPath();
            msgTime = GetTime();
            cout << msg << endl;
        }

        BeginDrawing();
        DrawChart(data, GetScreenWidth(), GetScreenHeight(), font);
        float u = GetScreenHeight() / 900.0f;
        const char* hint = (GetTime() - msgTime < 4) ? msg.c_str()
                                                     : "S: salvar PNG em alta resolução";
        DrawTextEx(font, hint, { 12 * u, GetScreenHeight() - 30 * u }, 18 * u, 0, GRAY);
        EndDrawing();
    }

    if (font.texture.id != GetFontDefault().texture.id) UnloadFont(font);
    CloseWindow();
    return 0;
}

// ============================================================================
//  Gera os arquivos finais: GIF do melhor jogando e PNG do gráfico
// ============================================================================
void ExportResults(const Individual& best, unsigned seed) {
    printf("Gravando GIF do melhor individuo...\n");
    Game g = RecordGif(best, seed, GifPath());
    printf("  %d comidas, %d passos, fim: %s\n", g.foods, g.steps, EndReasonText(g.reason));

    // A raylib precisa de uma janela para desenhar, então abrimos uma invisível
    printf("Gerando grafico...\n");
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(320, 200, "exportando");
    Font font = LoadChartFont();
    ExportChartPNG(ReadCsv(CsvPath()), font, PngPath());
    if (font.texture.id != GetFontDefault().texture.id) UnloadFont(font);
    CloseWindow();

    printf("\nArquivos gerados:\n  %s\n  %s\n  %s\n  %s\n",
           CsvPath().c_str(), BestPath().c_str(), GifPath().c_str(), PngPath().c_str());
    printf("\nPara ver a rede treinada jogando fases novas:  main.exe --jogar\n");
}

// ============================================================================
//  Modo --sem-janela: treino o mais rápido possível, só no terminal.
//  No fim, gera o gráfico (PNG) e a animação do melhor (GIF).
// ============================================================================
int RunHeadless(bool cont) {
    printf("Sensores: %s (%d entradas) | neuronios ocultos: %d | genes: %d\n",
           SENSORES_AVANCADOS ? "avancados" : "simples", N_INPUTS, N_HIDDEN, GENOME_SIZE);
    GeneticAlgorithm ga;
    ofstream csv;
    string err;
    if (!StartTraining(ga, cont, csv, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    double t0 = Now();
    while (true) {
        ga.EvaluateSome(1e9);
        GenStats s = ga.Stats();
        WriteCsvLine(csv, s);
        PrintStats(s);
        SaveBest(ga.pop[0], ga.generation + 1);
        SaveCheckpoint(ga);
        if (ga.generation + 1 >= NUM_GENERATIONS) break;
        ga.NextGeneration();
    }
    printf("Tempo total de treino: %.1f s\n\n", Now() - t0);
    csv.close();

    ExportResults(ga.pop[0], ga.seeds[0]);
    return 0;
}

// ============================================================================
//  Modo --exportar: gera PNG e GIF a partir dos arquivos já salvos
//  (por exemplo, depois de interromper um treino com Ctrl+C)
// ============================================================================
int RunExport() {
    Individual best;
    int gen = 0;
    string err;
    if (!LoadBest(BestPath(), best, gen, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    printf("Rede salva: geracao %d, fitness %.1f, %.1f comidas\n", gen, best.fitness, best.foods);

    // Usa a mesma fase da última geração salva, se houver checkpoint
    unsigned seed = 12345;
    GeneticAlgorithm ga;
    if (LoadCheckpoint(ga, err) && !ga.seeds.empty()) seed = ga.seeds[0];

    ExportResults(best, seed);
    return 0;
}

// ============================================================================
//  Modo --jogar: carrega a rede treinada e joga, SEM treinar.
//  Cada partida é uma fase nova (que a cobra nunca viu durante o treino).
// ============================================================================
int RunPlay(const string& path) {
    Individual brain;
    int gen = 0;
    string err;
    bool ok = LoadBest(path, brain, gen, err);
    if (!ok) cerr << err << endl;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(GAME_PX, GAME_PX, "Snake - jogando com a rede treinada");
    SetTargetFPS(60);

    random_device rd;
    mt19937 phases(rd()); // fases aleatórias de verdade
    Game game(phases());
    int    match = 1, finished = 0, record = 0;
    double sumFoods = 0;
    double lastStep = 0, stepInterval = 0.04, endTime = -1;

    while (!WindowShouldClose()) {
        if (ok) {
            if (IsKeyPressed(KEY_UP))   stepInterval = max(0.005, stepInterval / 1.5);
            if (IsKeyPressed(KEY_DOWN)) stepInterval = min(0.3,   stepInterval * 1.5);
            if (IsKeyPressed(KEY_R))    { game = Game(phases()); match++; }

            if (GetTime() - lastStep >= stepInterval) {
                lastStep = GetTime();
                if (game.alive) {
                    game.Step(ChooseDirection(brain.w, game));
                    if (!game.alive) {
                        endTime = GetTime();
                        finished++;
                        sumFoods += game.foods;
                        record = max(record, game.foods);
                    }
                } else if (GetTime() - endTime > 2.0) {
                    game = Game(phases());
                    match++;
                }
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        if (!ok) {
            DrawText(err.c_str(), 20, 20, 18, RED);
        } else {
            DrawGame(game);
            DrawText(TextFormat("Rede treinada (geracao %d) - partida %d", gen, match), 8, 8, 18, WHITE);
            DrawText(TextFormat("Comidas: %d   Passos: %d", game.foods, game.steps), 8, 30, 18, WHITE);
            if (finished > 0)
                DrawText(TextFormat("Recorde: %d   Media: %.1f comidas (%d partidas)",
                                    record, sumFoods / finished, finished), 8, 52, 18, GOLD);
            if (!game.alive) {
                const char* t = TextFormat("Fim: %s", EndReasonText(game.reason));
                int tw = MeasureText(t, 22);
                DrawRectangle(GAME_PX / 2 - tw / 2 - 12, GAME_PX / 2 - 22, tw + 24, 44, Fade(BLACK, 0.8f));
                DrawText(t, GAME_PX / 2 - tw / 2, GAME_PX / 2 - 11, 22, YELLOW);
            }
            DrawText("R: nova partida   Setas cima/baixo: velocidade", 8, GAME_PX - 24, 16, GRAY);
        }
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char** argv) {
    bool chart = false, headless = false, play = false, cont = false, exportOnly = false;
    string playPath = BestPath();
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if      (a == "--grafico")    chart = true;
        else if (a == "--sem-janela") headless = true;
        else if (a == "--continuar")  cont = true;
        else if (a == "--exportar")   exportOnly = true;
        else if (a == "--jogar") {
            play = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') playPath = argv[++i];
        } else {
            printf("Argumento desconhecido: %s\n\n"
                   "Uso:\n"
                   "  main.exe                          treino com janelas\n"
                   "  main.exe --sem-janela             treino so no terminal\n"
                   "  main.exe [--sem-janela] --continuar   continua o ultimo treino\n"
                   "  main.exe --grafico                so o grafico\n"
                   "  main.exe --exportar               gera PNG e GIF do que ja foi salvo\n"
                   "  main.exe --jogar [arquivo.txt]    joga com a rede treinada\n", a.c_str());
            return 1;
        }
    }
    MigrateOldResults();
    if (!chart) printf("Configuracao: %s\nResultados em: %s\n\n", ConfigName().c_str(), ResultDir().c_str());

    if (chart)      return RunChartWindow();
    if (exportOnly) return RunExport();
    if (play)     return RunPlay(playPath);
    if (headless) return RunHeadless(cont);
    return RunTraining(argv[0], cont);
}