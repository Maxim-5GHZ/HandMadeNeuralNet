#include "genetic.h"
#include "snake.hpp"
#include <iostream>
#include <ncurses.h>

template<typename T>
class SnakeTrainer {
private:
    Genetic<T>* genTrainer;
    int width;
    int height;
    size_t max_generations;
    size_t tournament_size;
    T mutation_rate;
    int max_steps;
    bool visualize;
    int target_score;

public:
    SnakeTrainer(Genetic<T>& gen, int w, int h, size_t max_gen, size_t tour_size, T mut_rate, int max_steps_per_game = 10000,bool vis = true, int target = 10)
        : genTrainer(&gen), width(w), height(h), max_generations(max_gen),
          tournament_size(tour_size), mutation_rate(mut_rate), 
          max_steps(max_steps_per_game), visualize(vis), target_score(target) {}

    void run() {
        size_t population_size = genTrainer->getPopulationSize();
        bool target_reached = false;
        
        if (visualize) {
            initscr();
            cbreak();
            noecho();
            curs_set(0);
            timeout(100);
            keypad(stdscr, TRUE);
        }

        for (size_t gen_num = 0; gen_num < max_generations && !target_reached; gen_num++) {
            T best_fitness = T(0);
            T total_fitness = T(0);
            size_t best_index = 0;

            for (size_t i = 0; i < population_size; i++) {
                SnakeGame game(width, height, max_steps);
                Perceptrone<T>& model = genTrainer->getModel(i);
                
                if (visualize && i == 0) { 
                    game.runWithRender(model);
                } else {
                    int score = game.runWithoutRender<T>(model);
                    T fitness = static_cast<T>(score);
                    
                    genTrainer->setFitness(i, fitness);
                    total_fitness += fitness;
                    
                    if (fitness > best_fitness) {
                        best_fitness = fitness;
                        best_index = i;
                    }

             
                    if (score >= target_score) {
                        target_reached = true;
                        break;
                    }
                }
            }

            if (visualize) {
                clear();
                mvprintw(0, 0, "Generation: %zu", gen_num + 1);
                mvprintw(1, 0, "Best score: %.1f", static_cast<double>(best_fitness));
                mvprintw(2, 0, "Avg score: %.1f", static_cast<double>(total_fitness / population_size));
                mvprintw(3, 0, "Target score: %d", target_score);
                refresh();
                napms(500); 
            } else {
                std::cout << "Generation " << gen_num + 1 
                          << ": Best = " << best_fitness 
                          << ", Avg = " << total_fitness / population_size 
                          << std::endl;
            }

            if (target_reached) {
                if (visualize) {
                    mvprintw(4, 0, "TARGET SCORE REACHED!");
                    refresh();
                    getch(); 
                } else {
                    std::cout << "TARGET SCORE REACHED!" << std::endl;
                }
                break;
            }

            genTrainer->tourSelect(tournament_size);
            
            for (size_t i = 0; i < population_size; i++) {
                if (i != best_index) {
                    genTrainer->mutate(i, mutation_rate);
                }
            }
        }

        if (visualize) {
            endwin();
        }
    }
};