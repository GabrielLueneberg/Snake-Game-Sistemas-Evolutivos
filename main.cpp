#include <iostream>
#include <raylib.h>
#include <deque>
#include <raymath.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

using namespace std;

#define TamPop 1000
#define GENOME_SIZE 200
#define TaxMut GENOME_SIZE / 10
#define MaxMoves GENOME_SIZE
#define NUM_GENERATIONS 200
#define LEVEL_SEED 12345

int gen = 0;
int cellSize = 30;
int cellCount = 36;
double lastUpdateTime = 0;

struct Individual
{
    vector<int> genome;
    int fitness;
};

bool ElementInDeque(Vector2 element, const deque<Vector2>& q) {
    for (unsigned int i = 0; i < q.size(); i++)
    {
        if (Vector2Equals(q[i], element))
        {
            return true;
        }
    }
    return false;
}

bool eventTriggered(double interval) {
   double currentTime = GetTime();
   if (currentTime - lastUpdateTime >= interval) {
      lastUpdateTime = currentTime;
      return true;
   }
   return false;
}

int ManhattanDistance(const Vector2& p1, const Vector2& p2) {
    return abs(static_cast<int>(p1.x - p2.x)) + abs(static_cast<int>(p1.y - p2.y));
}

class Food {
public:
    Vector2 position;

    Food(const deque<Vector2>& snakeBody) {
        position = RandPos(snakeBody);
    }

    Vector2 RandCell() {
        float x = GetRandomValue(0, cellCount - 1);
        float y = GetRandomValue(0, cellCount - 1);
        return Vector2{x, y};
    }

    Vector2 RandPos(const deque<Vector2>& snakeBody) {
        Vector2 pos = RandCell();
        while (ElementInDeque(pos, snakeBody)) {
            pos = RandCell();
        }
        return pos;
    }

    void Draw() {
        DrawRectangle(position.x * cellSize, position.y * cellSize, cellSize, cellSize, RED);
    }
};

class Snake {
public:
    deque<Vector2> body = {Vector2{6,9}, Vector2{5,9}, Vector2{4,9}};
    Vector2 direction = Vector2{1, 0};

    void Draw() {
        for (int i = 0; i < body.size(); i++) {
            if (i == 0) {
                DrawRectangle(body[i].x * cellSize, body[i].y * cellSize, cellSize, cellSize, BLUE);
            } else {
                DrawRectangle(body[i].x * cellSize, body[i].y * cellSize, cellSize, cellSize, GREEN);
            }
        }
    }

    void Update() {
        body.pop_back();
        body.push_front(Vector2{body[0].x + direction.x, body[0].y + direction.y});
    }

    void Reset() {
        body = {Vector2{6,9}, Vector2{5,9}, Vector2{4,9}};
        direction = Vector2{1, 0};
    }
};

class Game
{
public:
    Snake snake = Snake();
    Food food = Food(snake.body);
    bool running = true;

    void Draw()
    {
        food.Draw();
        snake.Draw();
    }

    void Update()
    {
        if (running)
        {
            snake.Update();
            CheckCollisionFood();
            CheckCollisionEdges();
            CheckCollisionSelf();
        }
    }

    bool CheckCollisionFood()
    {
        if (Vector2Equals(snake.body[0], food.position))
        {
            snake.body.push_back(snake.body.back());
            food.position = food.RandPos(snake.body);
            return true;
        }
        return false;
    }

    bool CheckCollisionEdges()
    {
        if (snake.body[0].x == cellCount || snake.body[0].x == -1)
        {
            GameOver();
            running = false;
            return true;
        }

        if (snake.body[0].y == cellCount || snake.body[0].y == -1)
        {
            GameOver();
            running = false;
            return true;
        }
        return false;
    }

    bool CheckCollisionSelf()
    {
        deque<Vector2> bodyWithoutHead = snake.body;
        bodyWithoutHead.pop_front();
        if (ElementInDeque(snake.body[0], bodyWithoutHead))
        {
            GameOver();
            running = false;
            return true;
        }
        return false;
    }

    void GameOver()
    {
        snake.Reset();
        food.position = food.RandPos(snake.body);
        running = false;
    }
};

class GeneticAlgorithm {
public:
    vector<Individual> population;

    void InitPop() {
        population.clear();
        for (int i = 0; i < TamPop; i++) {
            Individual ind;
            ind.fitness = 0;
            for (int j = 0; j < GENOME_SIZE; j++) {
                ind.genome.push_back(GetRandomValue(0, 3));
            }
            population.push_back(ind);

        }
    }

    void EvaluateFitness() {
        for (int i = 0; i < TamPop; i++) {
            SetRandomSeed(LEVEL_SEED);
            Game sim;
            population[i].fitness = 0;
            int moves = 0;
            int movesnofood = 0;

            while (sim.running && moves < MaxMoves) {
                int dir = population[i].genome[moves];
                ApplyDirection(sim.snake, dir);

                int initialSize = sim.snake.body.size();
                Vector2 foodbefore = sim.food.position;
                float distanceBefore = ManhattanDistance(sim.snake.body[0], foodbefore);
                
                sim.Update();

                // if ((int)sim.snake.body.size() > initialSize) {
                // population[i].fitness += 100;
                // movesnofood = 0;
                // }
        
                
            
                float distanceAfter = ManhattanDistance(sim.snake.body[0], foodbefore);

                if (distanceBefore - distanceAfter > 0) {
                    population[i].fitness += 1;
                } else if (distanceBefore - distanceAfter < 0) {
                    population[i].fitness -= 1;
                }
    
                


    

                moves++;
            }

            
        }

        sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    }

    int TournamentSelection() {
        int a = GetRandomValue(0, TamPop - 1);
        int b = GetRandomValue(0, TamPop - 1);
        return (population[a].fitness > population[b].fitness) ? a : b;
    }

    void NextGeneration() {
        vector<Individual> newPopulation;

        Individual elite = population[0];
        newPopulation.push_back(elite);

        while ((int)newPopulation.size() < TamPop) {
            int parent1Index = TournamentSelection();
            int parent2Index = TournamentSelection();

            Individual child;
            child.fitness = 0;
            child.genome.resize(GENOME_SIZE);
            int crossoverPoint = GetRandomValue(0, GENOME_SIZE - 1);
            for (int j = 0; j < GENOME_SIZE; j++) {
                child.genome[j] = (j<crossoverPoint)
                ? population[parent1Index].genome[j]
                : population[parent2Index].genome[j];
            }

            for (int k = 0; k < TaxMut; k++) {
                int segment = GetRandomValue(0, GENOME_SIZE - 1);
                child.genome[segment] = GetRandomValue(0, 3);
            }
            newPopulation.push_back(child);
        }

        population = newPopulation;
        gen++;
    }

    void ApplyDirection(Snake& snake, int dir) {
        if (dir == 0 && snake.direction.y != 1) snake.direction = Vector2{0, -1};
        else if (dir == 1 && snake.direction.y != -1) snake.direction = Vector2{0, 1};
        else if (dir == 2 && snake.direction.x != 1) snake.direction = Vector2{-1, 0};
        else if (dir == 3 && snake.direction.x != -1) snake.direction = Vector2{1, 0};
    }
};

int main() {
    GeneticAlgorithm ga;
    ga.InitPop();

    for (int i = 0; i < NUM_GENERATIONS; i++) {
        ga.EvaluateFitness();
        cout <<  ga.population[0].fitness << endl;

        if (i < NUM_GENERATIONS - 1) {
            ga.NextGeneration();
        }
    }

    InitWindow(cellCount * cellSize, cellCount * cellSize, "Snake Game - Genetico (melhor da geracao final)");
    SetTargetFPS(60);
    SetRandomSeed(LEVEL_SEED);
    Game displayGame;
    int currentMove = 0;

    while (WindowShouldClose() == false) {
        if (eventTriggered(0.05)) {
            if (currentMove < MaxMoves && displayGame.running) {
                ga.ApplyDirection(displayGame.snake, ga.population[0].genome[currentMove]);
                displayGame.Update();
                currentMove++;
            } else {
                displayGame.snake.Reset();
                displayGame.running = true;
                currentMove = 0;
                SetRandomSeed(LEVEL_SEED);
                displayGame.food = Food(displayGame.snake.body);
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        displayGame.Draw();
        DrawText(TextFormat("%d", ga.population[0].fitness), 10, 30, 20, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}