#include "geneticSnakeTrainer.hpp"

using namespace std;
using T = float;

int main() {
    const vector<size_t> neurons = {8, 16, 4};
    const vector<typename Activator<T>::Function> activate = {
        Activator<T>::RELU,

        Activator<T>::IDENTITY
    };
    
    Genetic<T> gen(neurons, activate, T(0.25), 20);
    SnakeTrainer<T> trainer(gen, 20, 20, 10, 3, 0.01, 10000, true, 100);
    trainer.run();
    
    return 0;
}