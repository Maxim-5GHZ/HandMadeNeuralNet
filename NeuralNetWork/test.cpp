#include<iostream>
#include"mlpActivators.hpp"

int main(){
    Activators activator({"relu","sigmoid","relu"});
    for(auto & i :activator.activation ){

        std::cout<< i(2)<<"\t";
    }
    return 0;
}