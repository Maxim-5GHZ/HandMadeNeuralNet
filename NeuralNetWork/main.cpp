#include <iostream>
#include <vector>
#include <cmath>
#include "mlp.h"
#include "exec_time.h"

using namespace std;

vector<float> normalize(const vector<float>& input) {
    vector<float> normalized = input;
    for (auto& val : normalized) {
        val /= 10.0;
    }
    return normalized;
}

vector<float> denormalize(const vector<float>& output) {
    vector<float> denormalized = output;
    for (auto& val : denormalized) {
        val *= 10.0;
    }
    return denormalized;
}

void quadratic_example() 
{
    MLP mlp(
    {
        1, 
        16, 
        16, 
        16,
        1}, 

    {   Activator::RELU,
        Activator::RELU,
        Activator::RELU,
        Activator::IDENTITY
    },
    0.25);

    vector<vector<float>> inputs;
    vector<vector<float>> targets;

    for (int x = 0; x <= 15; x++) {
        inputs.push_back({float(x)});
        targets.push_back({float(x*x)});
    }

    AppExecutionTimeCounter::StartMeasurement();

    int epochs = 100000;
    float learning_rate = 0.001;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = denormalize(mlp.train(normalize(inputs[i]), normalize(targets[i]), learning_rate));
            float error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= inputs.size();

        if (epoch % 10000 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }
    }

    float trainingTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время обучения (сек.): %1.3lf\n", trainingTimeSeconds);

    cout << "Результаты после обучения:" << endl;
    cout << "x   Сеть   Мат. Разность" << endl;

    AppExecutionTimeCounter::StartMeasurement();
    for (int x = 0; x <= 15; x ++) {
        auto output = denormalize(mlp.predict(normalize({float(x)})));
        printf("%3d %5.0lf %5d\t%2.0f\n", x, round(output[0]), x*x, float(x*x) - round(output[0]));
    }
    
    float predictTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время вычислений (мсек.): %1.3lf\n", predictTimeSeconds * 1000.0);

    mlp.save_weights("quadratic_weights.bin");
}

int main() {
    quadratic_example();
    return 0;
}