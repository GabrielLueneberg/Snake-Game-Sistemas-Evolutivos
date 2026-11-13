// ============================================================================
//  Snake Game + Algoritmo Genetico com controlador Perceptron (neuroevolucao)
//
//  - Cada individuo e uma rede neural (perceptron). O genoma sao os PESOS.
//  - A rede "enxerga" o jogo a cada passo (sensores) e decide: seguir reto,
//    virar a esquerda ou virar a direita.
//  - Evolucao darwiniana: variacao aleatoria (mutacao/crossover), heranca
//    (filhos herdam genes dos pais), selecao natural (torneio por fitness).
//    Nao ha aprendizado durante a vida (sem backpropagation / Lamarck).
//  - Mutacao avancada: gaussiana AUTO-ADAPTATIVA (Estrategias Evolutivas).
//    Cada individuo carrega no genoma sua propria "taxa de mutacao" (sigma),
//    que tambem e herdada e mutada; a selecao favorece indiretamente os
//    sigmas que geram bons filhos.
//  - Grafico: fitness do melhor individuo de cada geracao, em tempo real.
// ============================================================================

#include <raylib.h>
#include <array>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

// ----------------------------- Tabuleiro / janela ---------------------------
const int CELL_COUNT = 30;                     // celulas por lado
const int CELL_SIZE  = 22;                     // pixels por celula
const int GAME_PX    = CELL_COUNT * CELL_SIZE; // area do jogo (quadrada)
const int PANEL_W    = 560;                    // largura do painel do grafico

// ----------------------------- Jogo / avaliacao -----------------------------
const int MAX_STEPS      = 3000; // limite de passos por partida
const int STARVE_LIMIT   = 150;  // passos sem comer -> morre de fome (evita loops infinitos)
const int GAMES_PER_EVAL = 3;    // partidas por individuo (fases diferentes)

// ----------------------------- Rede neural ----------------------------------
// N_HIDDEN = 0  -> perceptron de uma camada (entradas ligadas direto as saidas)
// N_HIDDEN > 0  -> perceptron multicamadas (uma camada oculta com tanh)
const int N_INPUTS  = 8;
const int N_HIDDEN  = 0;
const int N_OUTPUTS = 3; // 0 = reto, 1 = esquerda, 2 = direita
const int GENOME_SIZE = (N_HIDDEN == 0)
    ? (N_INPUTS + 1) * N_OUTPUTS                               // +1 = bias
    : (N_INPUTS + 1) * N_HIDDEN + (N_HIDDEN + 1) * N_OUTPUTS;

// ----------------------------- Algoritmo genetico ---------------------------
const int      POP_SIZE        = 500;
const int      NUM_GENERATIONS = 200;
const int      ELITE           = 2;    // melhores que sobrevivem intactos
const int      TOURNAMENT_K    = 3;    // tamanho do torneio
const double   CROSSOVER_RATE  = 0.7;
const double   SIGMA_INIT      = 0.5;  // passo de mutacao inicial
const double   SIGMA_MIN       = 0.01;
const double   SIGMA_MAX       = 2.0;
const unsigned GA_SEED         = 2026; // semente do AG (troque para outro experimento)

// ============================================================================
//  Jogo
// ============================================================================
struct Cell { int x, y; };
bool operator==(Cell a, Cell b) { return a.x == b.x && a.y == b.y; }

// 0 = cima, 1 = direita, 2 = baixo, 3 = esquerda (sentido horario)
const Cell DIRS[4] = { {0, -1}, {1, 0}, {0, 1}, {-1, 0} };

int Manhattan(Cell a, Cell b) { return abs(a.x - b.x) + abs(a.y - b.y); }

class Game {
public:
    deque<Cell> body;
    int  dir;
    Cell food;
    bool alive;
    int  steps, foods, stepsSinceFood;
    mt19937 rng; // gerador proprio do jogo (independente do AG)
    vector<unsigned char> grid; // ocupacao do tabuleiro (consulta O(1))

    explicit Game(unsigned seed) : rng(seed) { Reset(); }

    void Reset() {
        int y = CELL_COUNT / 2;
        body  = { {6, y}, {5, y}, {4, y} };
        grid.assign(CELL_COUNT * CELL_COUNT, 0);
        for (Cell c : body) grid[c.y * CELL_COUNT + c.x]++;
        dir   = 1;
        alive = true;
        steps = foods = stepsSinceFood = 0;
        PlaceFood();
    }

    bool InBody(Cell c, bool ignoreTail) const {
        int n = grid[c.y * CELL_COUNT + c.x];
        if (ignoreTail && c == body.back()) n--;
        return n > 0;
    }

    // Parede ou corpo (o rabo sai do lugar no proximo passo, entao nao conta)
    bool IsBlocked(Cell c) const {
        if (c.x < 0 || c.y < 0 || c.x >= CELL_COUNT || c.y >= CELL_COUNT) return true;
        return InBody(c, true);
    }

    void PlaceFood() {
        uniform_int_distribution<int> d(0, CELL_COUNT - 1);
        do { food = { d(rng), d(rng) }; } while (InBody(food, false));
    }

    void Step(int newDir) {
        if (!alive) return;
        if ((newDir + 2) % 4 != dir) dir = newDir; // nao deixa inverter
        Cell head = { body[0].x + DIRS[dir].x, body[0].y + DIRS[dir].y };
        bool eat = (head == food);
        if (head.x < 0 || head.y < 0 || head.x >= CELL_COUNT || head.y >= CELL_COUNT) {
            alive = false;
            return;
        }
        if (!eat) { grid[body.back().y * CELL_COUNT + body.back().x]--; body.pop_back(); }
        if (InBody(head, false)) { alive = false; return; }
        body.push_front(head);
        grid[head.y * CELL_COUNT + head.x]++;
        steps++;
        stepsSinceFood++;

        if (eat) {
            foods++;
            stepsSinceFood = 0;
            if ((int)body.size() == CELL_COUNT * CELL_COUNT) { alive = false; return; } // zerou
            PlaceFood();
        }
        if (stepsSinceFood > STARVE_LIMIT) alive = false;
    }

    // ------------------------------------------------------------------------
    //  Sensores (no referencial da cobra: frente / esquerda / direita)
    //  [0..2] perigo imediato  (1 se a proxima casa e parede/corpo)
    //  [3..5] proximidade do obstaculo (1/distancia)
    //  [6]    comida a frente (+) ou atras (-)   normalizado em [-1, 1]
    //  [7]    comida a direita (+) ou esquerda (-) normalizado em [-1, 1]
    // ------------------------------------------------------------------------
    array<double, N_INPUTS> Sense() const {
        array<double, N_INPUTS> in{};
        int dS = dir, dL = (dir + 3) % 4, dR = (dir + 1) % 4;
        int rel[3] = { dS, dL, dR };
        Cell h = body[0];

        for (int k = 0; k < 3; k++) {
            Cell d = DIRS[rel[k]];
            Cell c = { h.x + d.x, h.y + d.y };
            int dist = 1;
            while (!IsBlocked(c)) { dist++; c.x += d.x; c.y += d.y; }
            in[k]     = (dist == 1) ? 1.0 : 0.0;
            in[3 + k] = 1.0 / dist;
        }

        int fx = food.x - h.x, fy = food.y - h.y;
        double len = abs(fx) + abs(fy);
        if (len == 0) len = 1;
        in[6] = (fx * DIRS[dS].x + fy * DIRS[dS].y) / len;
        in[7] = (fx * DIRS[dR].x + fy * DIRS[dR].y) / len;
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

    // Saida vencedora (winner-takes-all)
    int best = (int)(max_element(out.begin(), out.end()) - out.begin());
    if (best == 1) return (g.dir + 3) % 4; // esquerda
    if (best == 2) return (g.dir + 1) % 4; // direita
    return g.dir;                          // reto
}

// ============================================================================
//  Individuo e fitness
// ============================================================================
struct Individual {
    vector<double> w;     // pesos da rede (genes)
    double sigma = SIGMA_INIT; // passo de mutacao (tambem e um gene)
    double fitness = 0;
    double foods = 0;     // media de comidas por partida
};

// Uma partida. O individuo so e avaliado: nada da "experiencia" vai pro genoma.
double PlayGame(const vector<double>& w, unsigned seed, int& foodsOut) {
    Game g(seed);
    double fit = 0;
    while (g.alive && g.steps < MAX_STEPS) {
        Cell oldFood    = g.food;
        int  before     = Manhattan(g.body[0], oldFood);
        int  foodsBefore = g.foods;

        g.Step(ChooseDirection(w, g));

        if (!g.alive) { fit -= 20; break; }                 // morreu
        if (g.foods > foodsBefore) fit += 100;              // comeu
        else fit += (Manhattan(g.body[0], oldFood) < before) ? 1.0 : -1.5; // aproximou / afastou
    }
    foodsOut = g.foods;
    return fit;
}

// ============================================================================
//  Algoritmo genetico
// ============================================================================
class GeneticAlgorithm {
public:
    vector<Individual> pop;
    vector<unsigned>   seeds; // fases da geracao atual (iguais para todos)
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

    // O ambiente muda a cada geracao: sobrevive quem joga bem em qualquer fase,
    // e nao quem decorou um caminho.
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

    // Avalia individuos ate gastar 'budget' segundos (mantem a janela fluida).
    // Retorna true quando a geracao inteira foi avaliada.
    bool EvaluateSome(double budget) {
        double start = GetTime();
        while (evalIndex < pop.size()) {
            Evaluate(pop[evalIndex++]);
            if (GetTime() - start > budget) break;
        }
        if (evalIndex < pop.size()) return false;
        sort(pop.begin(), pop.end(),
             [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        return true;
    }

    // Selecao natural: torneio entre K individuos sorteados
    const Individual& Tournament() {
        int best = RandInt(0, POP_SIZE - 1);
        for (int k = 1; k < TOURNAMENT_K; k++) {
            int c = RandInt(0, POP_SIZE - 1);
            if (pop[c].fitness > pop[best].fitness) best = c;
        }
        return pop[best];
    }

    // Mutacao gaussiana auto-adaptativa (log-normal, Schwefel):
    //   sigma' = sigma * exp(tau * N(0,1))
    //   w_i'   = w_i + sigma' * N(0,1)
    // A mutacao e cega: nao usa o fitness nem "sabe" para onde melhorar.
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
        // de uma estacao). Eles serao reavaliados nas fases novas.
        for (int e = 0; e < ELITE; e++) next.push_back(pop[e]);

        while ((int)next.size() < POP_SIZE) {
            const Individual& p1 = Tournament();
            const Individual& p2 = Tournament();
            Individual child = p1;

            if (Uniform() < CROSSOVER_RATE) {
                // Crossover uniforme: cada gene vem de um dos pais (50%)
                for (int i = 0; i < GENOME_SIZE; i++)
                    if (Uniform() < 0.5) child.w[i] = p2.w[i];
                child.sigma = sqrt(p1.sigma * p2.sigma); // heranca do sigma
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
//  Desenho
// ============================================================================
void DrawGame(const Game& g) {
    DrawRectangle(0, 0, GAME_PX, GAME_PX, BLACK);
    DrawRectangle(g.food.x * CELL_SIZE, g.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE, RED);
    for (size_t i = 0; i < g.body.size(); i++)
        DrawRectangle(g.body[i].x * CELL_SIZE + 1, g.body[i].y * CELL_SIZE + 1,
                      CELL_SIZE - 2, CELL_SIZE - 2, i == 0 ? BLUE : GREEN);
}

void DrawChart(const vector<double>& hist, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, Color{ 25, 25, 35, 255 });
    DrawRectangleLines(x, y, w, h, GRAY);
    DrawText("Geracao", x + w / 2 - 35, y + h + 28, 16, LIGHTGRAY);

    if (hist.empty()) {
        DrawText("Avaliando a primeira geracao...", x + 20, y + h / 2, 18, LIGHTGRAY);
        return;
    }

    double lo = min(0.0, *min_element(hist.begin(), hist.end()));
    double hi = *max_element(hist.begin(), hist.end());
    if (hi - lo < 1) hi = lo + 1;
    double pad = (hi - lo) * 0.05;
    lo -= pad; hi += pad;

    auto PX = [&](int i)    { return x + (float)i / (NUM_GENERATIONS - 1) * w; };
    auto PY = [&](double v) { return y + h - (float)((v - lo) / (hi - lo)) * h; };

    // Grade e rotulos do eixo Y
    for (int k = 0; k <= 4; k++) {
        double v  = lo + (hi - lo) * k / 4.0;
        int    yy = (int)PY(v);
        DrawLine(x, yy, x + w, yy, Color{ 55, 55, 70, 255 });
        DrawText(TextFormat("%.0f", v), x - 60, yy - 7, 14, LIGHTGRAY);
    }
    // Rotulos do eixo X
    for (int g = 0; g < NUM_GENERATIONS; g += 50)
        DrawText(TextFormat("%d", g + 1), (int)PX(g) - 5, y + h + 6, 14, LIGHTGRAY);
    DrawText(TextFormat("%d", NUM_GENERATIONS), (int)PX(NUM_GENERATIONS - 1) - 12, y + h + 6, 14, LIGHTGRAY);

    // Linha do zero
    if (lo < 0 && hi > 0) DrawLine(x, (int)PY(0), x + w, (int)PY(0), DARKGRAY);

    // Curva do melhor individuo
    for (size_t i = 1; i < hist.size(); i++)
        DrawLineEx({ PX((int)i - 1), PY(hist[i - 1]) }, { PX((int)i), PY(hist[i]) }, 2.0f, GOLD);
    DrawCircleV({ PX((int)hist.size() - 1), PY(hist.back()) }, 4, ORANGE);
}

// ============================================================================
//  main
// ============================================================================
int main() {
    InitWindow(GAME_PX + PANEL_W, GAME_PX, "Snake - Perceptron + Algoritmo Genetico");
    SetTargetFPS(60);

    GeneticAlgorithm ga;
    ga.Init();

    ofstream csv("evolucao.csv");
    csv << "geracao,melhor_fitness,comidas_melhor,sigma_melhor\n";

    vector<double> history;      // fitness do melhor de cada geracao
    bool trainingDone = false;

    // Melhor da geracao mais recente (vai para a tela quando a partida atual acabar)
    Individual latestBest;
    unsigned   latestSeed = 0;
    int        latestGen  = -1;

    // O que esta sendo exibido agora
    Individual shown;
    int  shownGen   = -1;
    Game replay(0);
    bool hasReplay  = false;
    double lastStep = 0, stepInterval = 0.04;

    while (!WindowShouldClose()) {
        // ---------------- Treino (um pedaco por frame) ----------------
        if (!trainingDone && ga.EvaluateSome(0.012)) {
            const Individual& b = ga.pop[0];
            history.push_back(b.fitness);
            csv << ga.generation + 1 << "," << b.fitness << "," << b.foods << "," << b.sigma << "\n";
            csv.flush();
            cout << "Geracao " << ga.generation + 1 << "  melhor fitness: " << b.fitness
                 << "  comidas: " << b.foods << "  sigma: " << b.sigma << endl;

            latestBest = b;
            latestSeed = ga.seeds[0];
            latestGen  = ga.generation;

            if (ga.generation + 1 >= NUM_GENERATIONS) trainingDone = true;
            else ga.NextGeneration();
        }

        // ---------------- Exibicao do melhor individuo ----------------
        if (IsKeyPressed(KEY_UP))   stepInterval = max(0.005, stepInterval / 1.5);
        if (IsKeyPressed(KEY_DOWN)) stepInterval = min(0.3,   stepInterval * 1.5);

        if (!hasReplay && latestGen >= 0) {
            shown = latestBest; shownGen = latestGen;
            replay = Game(latestSeed); hasReplay = true;
        }
        if (hasReplay && GetTime() - lastStep >= stepInterval) {
            lastStep = GetTime();
            if (replay.alive && replay.steps < MAX_STEPS) {
                replay.Step(ChooseDirection(shown.w, replay));
            } else {
                // Partida acabou: passa a mostrar o melhor da geracao mais recente
                shown = latestBest; shownGen = latestGen;
                replay = Game(latestSeed);
            }
        }

        // ---------------- Desenho ----------------
        BeginDrawing();
        ClearBackground(Color{ 15, 15, 20, 255 });

        if (hasReplay) {
            DrawGame(replay);
            DrawText(TextFormat("Exibindo: melhor da geracao %d", shownGen + 1), 8, 8, 18, WHITE);
            DrawText(TextFormat("Comidas: %d   Passos: %d", replay.foods, replay.steps), 8, 30, 18, WHITE);
        } else {
            DrawRectangle(0, 0, GAME_PX, GAME_PX, BLACK);
            DrawText("Treinando...", GAME_PX / 2 - 60, GAME_PX / 2, 20, WHITE);
        }

        int px0 = GAME_PX;
        DrawRectangle(px0, 0, PANEL_W, GAME_PX, Color{ 20, 20, 28, 255 });
        DrawText("Fitness do melhor individuo por geracao", px0 + 20, 15, 20, RAYWHITE);
        DrawChart(history, px0 + 75, 55, PANEL_W - 100, GAME_PX - 250);

        int ty = GAME_PX - 140;
        DrawText(TextFormat("Geracao: %d / %d  %s", (int)history.size(), NUM_GENERATIONS,
                            trainingDone ? "(concluido)" : "(treinando)"), px0 + 20, ty, 18, WHITE);
        if (!history.empty()) {
            DrawText(TextFormat("Melhor fitness: %.1f", latestBest.fitness), px0 + 20, ty + 24, 18, GOLD);
            DrawText(TextFormat("Comidas do melhor (media): %.1f", latestBest.foods), px0 + 20, ty + 48, 18, WHITE);
            DrawText(TextFormat("Sigma de mutacao do melhor: %.3f", latestBest.sigma), px0 + 20, ty + 72, 18, WHITE);
        }
        DrawText("Setas cima/baixo: velocidade da exibicao", px0 + 20, ty + 104, 16, GRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}