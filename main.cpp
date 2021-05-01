//
//  main.cpp
//  OOP Final Project Neural Network
//
//  Created by Michael_hzy on 2019/12/9.
//  Copyright © 2019 Michael_hzy. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>
#include <typeinfo>
#include "Network.h"
#include "Matrix.h"


using namespace std;

int main()
{
    vector<vector<double>> X_train;
    vector<double> Y_train;
    double output{0};
    ifstream finy_train("/Users/michael_hzy/Desktop/Final Project Neural Network/Y_train.txt",ios::in);
    while(finy_train>>output)
    {
        Y_train.push_back(output);
    }
    ifstream finy_test("/Users/michael_hzy/Desktop/Final Project Neural Network/Y_test.txt",ios::in);
    vector<double> Y_test;
    while(finy_test>>output)
    {
        Y_test.push_back(output);
    }
    double dividend;
    double earnings;
    double cpi;
    double interastRate;
    double realPrice;
    double realDividend;
    double realEarnings;
    ifstream finx_train("/Users/michael_hzy/Desktop/Final Project Neural Network/X_train.txt",ios::in);
    while(finx_train>>dividend>>earnings>>cpi>>interastRate>>realPrice>>realDividend>>realEarnings){
        vector<double> input;
        input.push_back(dividend);
        input.push_back(earnings);
        input.push_back(cpi);
        input.push_back(interastRate);
        input.push_back(realPrice);
        input.push_back(realDividend);
        input.push_back(realEarnings);
        X_train.push_back(input);
    }
    vector<vector<double>> X_test;
    ifstream finx_test("/Users/michael_hzy/Desktop/Final Project Neural Network/X_test.txt",ios::in);
    while(finx_test>>dividend>>earnings>>cpi>>interastRate>>realPrice>>realDividend>>realEarnings){
        vector<double> input;
        input.push_back(dividend);
        input.push_back(earnings);
        input.push_back(cpi);
        input.push_back(interastRate);
        input.push_back(realPrice);
        input.push_back(realDividend);
        input.push_back(realEarnings);
        X_test.push_back(input);
    }
    vector<int> layers{7,10,10,10,10,1};
    int cycles{100};
    Network networkTrain(layers, X_train, Y_train,0);
    double toleranceEerr{0.005};
    double averageErr{100};
    int cycle{0};
    while (cycle<cycles&&averageErr>toleranceEerr)
    {
        for (int i = 0; i<1664; i++)
        {
            networkTrain.forwardPropagation(i);
            networkTrain.backwordPropagation(i);
            networkTrain.accumulateError();
            networkTrain.updateWeights(0.05);
        }
        averageErr = networkTrain.getErrors() /1500;
        cout<<"Epoch "<<setw(3)<<cycle<<" :  "<<"Average errors: "<<averageErr<<endl;
        networkTrain.setErrors(0);//reset the sum error to 0
        // for each cycle to accumulate the error and then calculate the average error for all cycles
        cycle++;
        
    }
    //get the final weights
    cout<<"The final weights matrix is: "<<endl;
    for(auto j:networkTrain.getWeights()){
               cout<<j<<endl;
           }
    //build the test network and predict
    Network networkTest(layers, X_test, Y_test,0);
    networkTest.setWeights(networkTrain.getWeights());
    networkTest.setBias(networkTrain.getBias());
    cout<<"now start testing\n";
    double testErrors{0};
    for (int i{0}; i<100; i++)
    {
        networkTest.forwardPropagation(i);
        cout<<"real:"<<setw(10)<<Y_test[i]<<"\tpredict:"<<networkTest.getLayerOutputs().rbegin()->getData()[0][0]<<endl;
        testErrors+=pow((networkTest.getLayerOutputs().rbegin()->getData()[0][0] -Y_test[i]),2);
        networkTest.clearLayerOutput();
    }
    cout<<"The test average errors:"<<testErrors/100<<endl;
}

