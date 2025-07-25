#include <iostream>
#include "genetic.h"

using namespace std;

template<typename T>
vector<T> normalize(const vector<T>& input) {
    vector<T> normalized = input;
    for (auto& val : normalized) {
        val /= T(10);
    }
    return normalized;
}

template<typename T>
vector<T> denormalize(const vector<T>& output) {
    vector<T> denormalized = output;
    for (auto& val : denormalized) {
        val *= T(10);
    }
    return denormalized;
}

int main() {
    using T = float;
    size_t populationSize = 500;
    Genetic<T> mlp(
        {1, 32,32 ,1}, 
        {   Activator<T>::RELU,
            Activator<T>::RELU,
            Activator<T>::IDENTITY
        },
        T(0.40),
        populationSize
    );

    int xx = 5;
    vector<vector<T>> inputs;
    vector<vector<T>> targets;

  
    for (int x = 0; x <= xx; x++) {
        inputs.push_back({T(x)});
        targets.push_back({T(x*x)});
    }

    T total_error = T(1000000);

    const T mutation_rate = 0.001;
    const T target_error = 0.001;
    
    for (int epoch = 0; total_error > target_error; ++epoch) {
        total_error = T(0);
        
        for (size_t numberModel = 0; numberModel < populationSize; numberModel++) {
            T model_error = T(0);
            auto& model = mlp.getModel(numberModel);
            
            for (size_t i = 0; i < inputs.size(); i++) {
                vector<T> prediction = model.predict(inputs[i]);
                model_error += abs(targets[i][0] - prediction[0]);
            }
            
            mlp.setFitness(numberModel, 1.0 / (1.0 + model_error)); 
            total_error += model_error;
        }
        
        total_error /= populationSize * inputs.size();
        
        
        mlp.tourSelect(10); 
        for (size_t i = 0; i < populationSize; i++) {
            mlp.mutate(i, mutation_rate);
        }
        
        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << ", Error: " << total_error << endl;
        }
    }

    auto& best_model = mlp.getModel(0);
    cout << "\nTesting trained model:\n";
    for (int x = 0; x <= xx; x++) {
        vector<T> input_norm = {T(x)};
        T predicted_norm = best_model.predict(input_norm)[0];
        T predicted = {predicted_norm};
        T actual = x * x;
        cout<<"Ввод:" << x << " Предсказание:" << predicted 
             << " Реальное значение: " << actual<< endl;
    }

    return 0;
}