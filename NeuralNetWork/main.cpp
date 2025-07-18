#include <iostream>
#include <vector>
#include <cmath>
#include "mlp.h"

using namespace std;

vector<double> normalize(const vector<double>& input) {
    vector<double> normalized = input;
    for (auto& val : normalized) {
        val /= 10.0;
    }
    return normalized;
}


vector<double> denormalize(const vector<double>& output) {
    vector<double> denormalized = output;
    for (auto& val : denormalized) {
        val *= 10.0;
    }
    return denormalized;
}


void quadratic_example() {

    vector<size_t> neurons = {1, 16, 16, 1};
    vector<string> activations = {"leaky_relu", "leaky_relu", "identity"};
    MLP mlp(neurons, activations);

    vector<vector<double>> inputs;
    vector<vector<double>> targets;

    for (int x = 0; x <= 15; x++) {
        inputs.push_back({double(x)});
        targets.push_back({double(x*x)});
    }

    int epochs = 1000000;
    double learning_rate = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = denormalize(mlp.train(normalize(inputs[i]), normalize(targets[i]), learning_rate));
            double error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= inputs.size();

        if (epoch % 1000 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }
    }

    cout << "\nРезультаты после обучения:" << endl;
    cout << "x\tПрогноз\tРеальное значение" << endl;

    for (int x = 0; x <= 15; x ++) {
        auto output = denormalize(mlp.predict(normalize({double(x)})));
        cout << x << "\t" << output[0] << "\t" << x*x << endl;
    }

    mlp.save_weights("quadratic_weights.bin");
   
}

int main() {
   
    quadratic_example();

    return 0;
}
