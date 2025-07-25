#include "genetic.h"
#include <algorithm>
#include <numeric>

template<typename T>
Genetic<T>::Genetic(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue, size_t populationSize) : gen(std::random_device{}()) {
    generations.reserve(populationSize);
    for(size_t i = 0; i < populationSize; i++) {
        generations.emplace_back(Perceptrone<T>(neurons, activate, maxBiasValue));
    }
}

template<typename T>
Perceptrone<T>& Genetic<T>::getModel(size_t numModel) {
    return generations[numModel].model;
}

template<typename T>
const Perceptrone<T>& Genetic<T>::getModel(size_t numModel) const {
    return generations[numModel].model;
}

template<typename T>
void Genetic<T>::setFitness(size_t numModel, T fitness) {
    generations[numModel].fitness = fitness;
}

template<typename T>
void Genetic<T>::mutate(Gen& gen, T mutationRate) {
    std::uniform_real_distribution<T> prob_dist(0, 1);
    std::normal_distribution<T> noise_dist(0, 0.1);

    auto weights = gen.model.get_weights();
    auto biases = gen.model.get_biases();

 
    for (auto& layer : weights) {
        for (auto& neuron_weights : layer) {
            for (auto& w : neuron_weights) {
                if (prob_dist(this->gen) < mutationRate) {
                    w += noise_dist(this->gen);
                }
            }
        }
    }

    
    for (auto& layer : biases) {
        for (auto& b : layer) {
            if (prob_dist(this->gen) < mutationRate) {
                b += noise_dist(this->gen);
            }
        }
    }

    gen.model.set_weights(weights);
    gen.model.set_biases(biases);
}

template<typename T>
void Genetic<T>::mutate(size_t index, T mutationRate) {
    mutate(generations[index], mutationRate);
}

template<typename T>
void Genetic<T>::tourSelect(size_t tournamentSize) {
    std::vector<Gen> new_generation;
    new_generation.reserve(generations.size());
    std::uniform_int_distribution<size_t> dist(0, generations.size() - 1);
    
    for (size_t i = 0; i < generations.size(); i++) {
        size_t best_index = dist(gen);
        T best_fitness = generations[best_index].fitness;
        
        for (size_t j = 1; j < tournamentSize; j++) {
            size_t candidate_index = dist(gen);
            if (generations[candidate_index].fitness > best_fitness) {
                best_index = candidate_index;
                best_fitness = generations[candidate_index].fitness;
            }
        }
        new_generation.push_back(generations[best_index]);
    }
    
    generations = std::move(new_generation);
}

template<typename T>
void Genetic<T>::rouletteSelect() {
    std::vector<T> fitnesses;
    fitnesses.reserve(generations.size());
    for (const auto& g : generations) {
        fitnesses.push_back(g.fitness);
    }
    
    T sum_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), T(0));
    std::uniform_real_distribution<T> dist(0, sum_fitness);
    
    std::vector<Gen> new_generation;
    new_generation.reserve(generations.size());
    
    for (size_t i = 0; i < generations.size(); i++) {
        T r = dist(gen);
        T running_sum = T(0);
        for (size_t j = 0; j < generations.size(); j++) {
            running_sum += fitnesses[j];
            if (running_sum >= r) {
                new_generation.push_back(generations[j]);
                break;
            }
        }
    }
    
    generations = std::move(new_generation);
}

template class Genetic<float>;
template class Genetic<double>;