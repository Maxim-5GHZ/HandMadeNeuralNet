#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "mlp.h"

using namespace std;

vector<vector<double>> read_csv(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<double> row;
        int col = 0;
        double num1, num2, result;
        char op;

        while (getline(ss, token, ';')) {
            if (col == 0) num1 = stod(token);
            else if (col == 1) op = token[0];
            else if (col == 2) num2 = stod(token);
            else if (col == 3) result = stod(token);
            col++;
        }

        // Кодируем операцию в числовом виде (например, + = 0.1, - = 0.2, * = 0.3, / = 0.4)
        double op_encoded;
        switch(op) {
            case '+': op_encoded = 0.1; break;
            case '-': op_encoded = 0.2; break;
            case '*': op_encoded = 0.3; break;
            case '/': op_encoded = 0.4; break;
            default: op_encoded = 0.0;
        }

        data.push_back({num1, op_encoded, num2, result});
    }

    return data;
}



using namespace std;

int main() {

    auto dataset = read_csv("math_combinations.csv");
    
    
    vector<vector<double>> inputs;
    vector<vector<double>> targets;
    
    for (const auto& row : dataset) {
        inputs.push_back({row[0], row[1], row[2]}); // num1, op, num2
        targets.push_back({row[3]});                // result
    }


   
    vector<size_t> neurons = {3,16,1};
    

    vector<string> activations = { "elu","sigmoid"};


    MLP mlp(neurons, activations);

    


    
    int epochs = 100000;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            
            mlp.train(inputs[i], targets[i]);
            vector<double> output = mlp.prdct(inputs[i]);
            double error = output[0] - targets[i][0];
            total_error += error * error;
        }
        total_error /= inputs.size();

        if (epoch % 100 == 0) {
            cout << "Epoch: " << epoch << ", Error: " << total_error << endl;
        }
    }

  
    cout << "\nTest:\n";
    for (int i = 0;i<inputs.size();i++) {
        vector<double> input = {1,0.1,1};
        vector<double> output = mlp.prdct(input);
        cout << "Input: [" << input[0] << "] -> Output: " << output[0] << endl;
    }

    return 0;
}