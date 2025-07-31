#include<iostream>
#include"snake.hpp"
using namespace std;
using T =float;

int main(){
     SnakeConfig snake_config;
    snake_config.width = 5;
    snake_config.height = 5;
    snake_config.initial_length = 1;
    snake_config.max_steps = 100;
    snake_config.food_score = 1;
    snake_config.head_char = '@';
    snake_config.body_char = 'O';
    snake_config.food_char = '*';
    snake_config.wall_char = '#';
    snake_config.max_steps_without_food = 20;

    SnakeGame game(snake_config);


    const vector<typename Activator<T>::Function> activations = {
        Activator<T>::RELU,  
        Activator<T>::RELU, 
        Activator<T>::RELU,    
        Activator<T>::IDENTITY 
    };

    const vector<size_t> neurons = {8,64, 64, 64,4};


    Perceptrone<T> model(neurons,activations,0.3);

    model.load_weights("weights.bin");

    int score = game.runWithRender(model);

    cout<<endl<<score<<endl;
    
    return 0;
}