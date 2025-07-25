#include"genetic.h"

template<typename T>

Genetic<T>::Genetic(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue,size_t populationSize){

    for(size_t i = 0;i<populationSize;i++){
        Perceptrone<T> newPerceptron(neurons, activate, maxBiasValue);
        generations[0].model[i].push_back(newPerceptron);  
    }

}



template<typename T>

Perceptrone<T> Genetic<T>::getModel(size_t numModel,size_t numGen){
    return generations[numGen].model[numModel];
}


template<typename T>

void Genetic<T>::setFitness(size_t numModel,size_t numGen,T fitness){
    generations[numGen].fitnessk[numModel] = fitness
}
