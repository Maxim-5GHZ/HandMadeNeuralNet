#include <iostream>
#include <vector>
#include <cmath>
#include "mlp.h"
#include "exec_time.h"

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

void quadratic_example() 
{
    using T = float;
    MLP<T> mlp(
        {1, 64, 64,64,1}, 
        {
            
            Activator<T>::RELU,
            Activator<T>::RELU,
            Activator<T>::RELU,
            Activator<T>::IDENTITY
        },
        T(0.25)
    );
    int xx= 50;
    vector<vector<T>> inputs;
    vector<vector<T>> targets;

    for (int x = 0; x <= xx; x++) {
        inputs.push_back({T(x)});
        targets.push_back({T(x*x)});
    }

    AppExecutionTimeCounter::StartMeasurement();

    int epochs = 100000;
    T learning_rate = T(0.0000005);
    T total_error = T(1);
    for (int epoch = 0; total_error > 0.1; ++epoch) {
        total_error = T(0);

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = denormalize<T>(
                mlp.train(normalize<T>(inputs[i]), 
                normalize<T>(targets[i]), 
                learning_rate)
            );
            T error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= static_cast<T>(inputs.size());

        if (epoch % 10 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }
    }

    double trainingTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время обучения (сек.): %1.3lf\n", trainingTimeSeconds);

    cout << "Результаты после обучения:" << endl;
    cout << "x   Сеть   Мат. Разность" << endl;

    AppExecutionTimeCounter::StartMeasurement();
    for (int x = 0; x <= xx; x++) {
        auto output = denormalize<T>(mlp.predict(normalize<T>({T(x)})));
        T predicted = round(output[0]);
        T actual = T(x*x);
        printf("%3d %5.0f %5.0f\t%2.0f\n", 
               x, 
               static_cast<double>(predicted), 
               static_cast<double>(actual),
               static_cast<double>(actual - predicted));
    }
    
    double predictTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время вычислений (мсек.): %1.3lf\n", predictTimeSeconds * 1000.0);

    mlp.save_weights("quadratic_weights.bin");
}

int main() {
    quadratic_example();
    return 0;
}