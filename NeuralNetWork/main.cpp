#include <iostream>
#include <vector>
#include <cmath>
#include "mlp.h"

using namespace std;

// Функция для нормализации данных (пример)
vector<double> normalize(const vector<double>& input) {
    vector<double> normalized = input;
    // Простая нормализация делением на 10
    for (auto& val : normalized) {
        val /= 10.0;
    }
    return normalized;
}

// Функция для денормализации данных (пример)
vector<double> denormalize(const vector<double>& output) {
    vector<double> denormalized = output;
    for (auto& val : denormalized) {
        val *= 10.0;
    }
    return denormalized;
}

// Пример обучения сети на функции XOR
void xor_example() {
    cout << "\n=== Пример обучения на XOR ===" << endl;

    // Архитектура сети: 2 входа, скрытый слой с 3 нейронами, 1 выход
    vector<size_t> neurons = {2, 3, 1};
    vector<string> activations = {"relu", "sigmoid"}; // Активации для скрытого и выходного слоев

    // Создаем сеть
    MLP mlp(neurons, activations);

    // Данные для обучения (XOR)
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    // Обучение
    int epochs = 10000;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = mlp.train(inputs[i], targets[i], learning_rate);
            double error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= inputs.size();

        if (epoch % 1000 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }
    }

    // Тестирование
    cout << "\nРезультаты после обучения:" << endl;
    for (const auto& input : inputs) {
        auto output = mlp.predict(input);
        cout << input[0] << " XOR " << input[1] << " = " << output[0]
             << " (ожидалось: " << ((input[0] != input[1]) ? 1 : 0) << ")" << endl;
    }
}

// Пример обучения на квадратичной функции
void quadratic_example() {
    cout << "\n=== Пример обучения на квадратичной функции ===" << endl;

    // Архитектура сети: 1 вход, 2 скрытых слоя по 5 нейронов, 1 выход
    vector<size_t> neurons = {1, 5, 5, 1};
    vector<string> activations = {"relu", "relu", "linear"}; // Активации

    // Создаем сеть
    MLP mlp(neurons, activations);

    // Генерация данных для обучения (y = x^2)
    vector<vector<double>> inputs;
    vector<vector<double>> targets;

    for (double x = 0.0; x <= 1.0; x += 0.1) {
        inputs.push_back({x});
        targets.push_back({x*x});
    }

    // Обучение
    int epochs = 5000;
    double learning_rate = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = mlp.train(inputs[i], targets[i], learning_rate);
            double error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= inputs.size();

        if (epoch % 1000 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }
    }

    // Тестирование
    cout << "\nРезультаты после обучения:" << endl;
    cout << "x\tПрогноз\tРеальное значение" << endl;
    for (double x = 0.0; x <= 1.0; x += 0.2) {
        auto output = mlp.predict({x});
        cout << x << "\t" << output[0] << "\t" << x*x << endl;
    }

    // Сохраняем веса
    mlp.save_weights("quadratic_weights.bin");
    cout << "\nВеса сохранены в quadratic_weights.bin" << endl;

    // Загружаем веса в новую сеть
    MLP mlp2(neurons, activations);
    mlp2.load_weights("quadratic_weights.bin");
    cout << "Веса загружены в новую сеть" << endl;

    // Проверяем загруженную сеть
    cout << "\nПроверка загруженной сети (x=0.5): " << mlp2.predict({0.5})[0]
         << " (ожидалось: 0.25)" << endl;
}

int main() {
    // Пример с XOR
    xor_example();

    // Пример с квадратичной функцией
    quadratic_example();

    return 0;
}
