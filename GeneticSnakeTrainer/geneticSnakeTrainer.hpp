#ifndef GENETIC_SNAKE_TRAINER_HPP
#define GENETIC_SNAKE_TRAINER_HPP

#include "genetic.h"
#include "snake.hpp"
#include <iostream>
#include <ncurses.h>

template<typename T>
struct GeneticSnakeTrainerConfig {
    size_t max_generations = 100;
    size_t tournament_size = 3;
    T mutation_rate = 0.001;
    int max_steps_per_game = 10000;
    bool visualize = true;
    int target_score = 100;
    std::function<void(size_t, T, T)> on_generation_end = [](size_t, T, T){};
    std::function<void()> on_target_reached = [](){};
    SnakeConfig snake_config;
};

template<typename T>
class SnakeTrainer {
private:
    Genetic<T>* genTrainer;
    GeneticSnakeTrainerConfig<T> config;
    
public:
    SnakeTrainer(Genetic<T>& gen, const GeneticSnakeTrainerConfig<T>& cfg = {})
        : genTrainer(&gen), config(cfg) {}

    void run() {
        size_t population_size = genTrainer->getPopulationSize();
        bool target_reached = false;
        
        if (config.visualize) {
            initscr();
            cbreak();
            noecho();
            curs_set(0);
            timeout(100);
            keypad(stdscr, TRUE);
        }

        for (size_t gen_num = 0; gen_num < config.max_generations && !target_reached; gen_num++) {
            T best_fitness = T(0);
            T total_fitness = T(0);
            size_t best_index = 0;

            for (size_t i = 0; i < population_size; i++) {
                SnakeGame game(config.snake_config);
                Perceptrone<T>& model = genTrainer->getModel(i);
                
                if (config.visualize && i == 0) { 
                    int score = game.runWithRender(model);
                    T fitness = static_cast<T>(score);
                    genTrainer->setFitness(i, fitness);
                    total_fitness += fitness;
                    
                    if (fitness > best_fitness) {
                        best_fitness = fitness;
                        best_index = i;
                    }
                } else {
                    int score = game.runWithoutRender(model);
                    T fitness = static_cast<T>(score);
                    genTrainer->setFitness(i, fitness);
                    total_fitness += fitness;
                    
                    if (fitness > best_fitness) {
                        best_fitness = fitness;
                        best_index = i;
                    }

                    if (score >= config.target_score) {
                        target_reached = true;
                        break;
                    }
                }
            }

            if (config.visualize) {
                clear();
                mvprintw(0, 0, "Generation: %zu", gen_num + 1);
                mvprintw(1, 0, "Best score: %.1f", static_cast<double>(best_fitness));
                mvprintw(2, 0, "Avg score: %.1f", 
                        static_cast<double>(total_fitness / population_size));
                mvprintw(3, 0, "Target score: %d", config.target_score);
                refresh();
                napms(500);
            }

            config.on_generation_end(gen_num + 1, best_fitness, 
                                   total_fitness / population_size);

            if (target_reached) {
                if (config.visualize) {
                    mvprintw(4, 0, "TARGET SCORE REACHED!");
                    refresh();
                    getch();
                }
                config.on_target_reached();
                break;
            }

            genTrainer->tourSelect(config.tournament_size);
            
            for (size_t i = 0; i < population_size; i++) {
                if (i != best_index) {
                    genTrainer->mutate(i, config.mutation_rate);
                }
            }
        }

        if (config.visualize) {
            endwin();
        }
    }
};

#endif