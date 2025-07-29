#include "geneticSnakeTrainer.hpp"
#include <iostream>

using namespace std;
using T = float;

void printProgress(size_t gen, T best, T avg) {
    cout << "Generation " << gen 
         << ": Best = " << best 
         << ", Avg = " << avg << endl;
}

void onTargetReached() {
    cout << "\n=== TARGET SCORE REACHED! ===\n";
}

int main() {
    
    const vector<size_t> neurons = {8,32, 32, 32,4}; 
    const vector<typename Activator<T>::Function> activations = {
        Activator<T>::RELU,  
        Activator<T>::RELU, 
        Activator<T>::RELU,    
        Activator<T>::IDENTITY 
    };
    

    SnakeConfig snake_config;
    snake_config.width = 10;
    snake_config.height = 10;
    snake_config.initial_length = 5;
    snake_config.max_steps = 500;
    int max_steps_without_food = 25;
    snake_config.food_score = 1;
    snake_config.head_char = '@';
    snake_config.body_char = 'O';
    snake_config.food_char = '*';
    snake_config.wall_char = '#';
    
    GeneticSnakeTrainerConfig<T> trainer_config;
    trainer_config.max_generations = 1000;
    trainer_config.tournament_size = 50;
    trainer_config.mutation_rate = 0.01;
    trainer_config.target_score = 2000;
    trainer_config.visualize = 0;
    trainer_config.snake_config = snake_config;
    
  
    trainer_config.on_generation_end = &printProgress;
    trainer_config.on_target_reached = &onTargetReached;
    
    try {
        Genetic<T> genetic_algorithm(neurons, activations, T(0.25), 500);
        SnakeTrainer<T> trainer(genetic_algorithm, trainer_config);
        
        trainer.run();
        
        cout << "\nTraining completed!\n";
    } catch (...) {
        return 1;
    }
    
    return 0;
}