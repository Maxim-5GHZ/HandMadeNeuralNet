#include <vector>
#include <iostream>
#include <random>


using namespace std;


class MLP {

    vector<vector<double>> bias;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> data;
    
    void calculate();
    
public:
    MLP(const vector<size_t>& neurons);

    vector<double>prdct(vector<double>&input);



};