//
//  Network.cpp
//  OOP Final Project Neural Network
//
//  Created by Michael_hzy on 2019/12/9.
//  Copyright © 2019 Michael_hzy. All rights reserved.
//

#include "Network.h"
#include "Matrix.h"
#include<string>
#include<vector>
#include<iterator>
#include <cmath>
#include<cstdlib>
#include<ctime>
using namespace std;

Network::Network(vector<int> layer, vector<vector<double>>input, vector<double>output, double error) :layers{ layer }, inputData{ input }, outputData{ output }, errors(error)
{
    srand(static_cast<int>(time(0)));
    for (size_t i{ 0 }; i < layers.size() - 1; i++) {    // generate n-1 matrixs
        Matrix tempMatrix{ layers[i + 1],layers[i] };
        vector<double> initialWeight;
        for (int j{ 0 }; j < layers[i] * layers[i + 1]; j++) {
            initialWeight.push_back(rand() / double(RAND_MAX));
        }
        tempMatrix.setData(initialWeight, tempMatrix);
        weights.push_back(tempMatrix);
        initialWeight.clear();
    }
    for (size_t i{ 0 }; i < layers.size() - 1; i++) {
        Matrix tempMatrix{ layers[i + 1],1 };
        vector<double> initialbias;
        for (auto j{ 0 }; j < layers[i + 1]; j++) {
            initialbias.push_back(rand() / double(RAND_MAX));
        }
        tempMatrix.setData(initialbias, tempMatrix);
        bias.push_back(tempMatrix);
        initialbias.clear();
    }
}

vector<vector<double>> Network::getInputData()const {
    return inputData;
}

vector<double> Network::getOutputData() const {
    return outputData;
}

vector<Matrix> Network::getWeights() const {
    return weights;
}

void Network::setWeights(vector<Matrix> weightsMatrix){
    weights=weightsMatrix;
}

vector<Matrix> Network::getBias() const {
    return bias;
}

void Network::setBias(std::vector<Matrix> biasMatrix){
    bias=biasMatrix;
}

void Network::clearLayerOutput(){
    layerOutputs.clear();
}
vector<int> Network::getLayers() const {
    return layers;
}

double Network::getErrors()const {
    return errors;
}


void Network::setErrors(double tempError){
    errors=tempError;
}

double getPredict(){ //getYpredict
    return 0;
}

vector<Matrix> Network::getLayerOutputs()const{
    return layerOutputs;
}

void Network::forwardPropagation(int count) {
    Matrix X(layers[0], 1);  //the input data is n by 1 matrix
    Matrix::setData(inputData[count], X);//transform input data to a matrix
    Matrix tempout = X;
    Matrix tempin;
    layerOutputs.push_back(X);
    for (size_t i{ 0 }; i < layers.size() - 1; i++)
    {
        tempin = weights[i] * tempout + bias[i];
        layerInputs.push_back(tempin);
        tempout = Matrix::sigmod(tempin);//output for each layer
        layerOutputs.push_back(tempout);//push back each layers output resluts
    }
}

void Network::backwordPropagation(int count) {
    Matrix deltaY(1, 1);
    vector<double> Yact;//y actual value
    Yact.push_back(outputData[count] - (layerOutputs.rbegin())->getData()[0][0]); //Y'-Y
    outputErrors.push_back(outputData[count] - (layerOutputs.rbegin())->getData()[0][0]);
    Matrix::setData(Yact, deltaY);// set Yact's value to deltaY
    deltaY=deltaY*Matrix::sigmodDerivative(*(layerInputs.rbegin())); //(Y'-Y)*sigmod(Yin)
    errorVector.push_back(deltaY);
    Matrix deltaTemp = deltaY;
    for (size_t i{ 0 }; i < layers.size() - 2; ++i)
    {
        deltaTemp = (weights.rbegin() + i)->getTranspose() * deltaTemp.hadamard(Matrix::sigmodDerivative(*(layerInputs.rbegin() + i)));
        errorVector.push_back(deltaTemp);
    }
}

double Network::accumulateError()
{
    errors+=pow((*outputErrors.rbegin()),2);
    return errors;
}

void Network::updateWeights(double rate) {
    for(size_t i{ 0 }; i < layers.size() - 1; i++)
    {
        Matrix deltaWeights{ layers[i + 1],layers[i] };;
        deltaWeights= *(errorVector.rbegin()+i)*(layerOutputs[i].getTranspose())*rate;
        weights[i] = weights[i] + deltaWeights;
        
        Matrix deltaBias{ layers[i + 1],1 };
        deltaBias = (*(errorVector.rbegin()+i))* rate;
        bias[i] = bias[i] + deltaBias;
    }

    errorVector.clear();
    layerInputs.clear();
    layerOutputs.clear();
}

string Network::toString(){
    return "abc";
}


Network::~Network(){}
