//
//  Network.h
//  OOP Final Project Neural Network
//
//  Created by Michael_hzy on 2019/12/9.
//  Copyright © 2019 Michael_hzy. All rights reserved.
//

#ifndef Network_h
#define Network_h
#include"Matrix.h"
#include<vector>
#include<string>
class Network
{
public:
    /*layer[10,20,20,20,1] 10 in, 3*20 hidden, 1 out
      input m by n vector,
      output column matrix
      error
      cycle
    */
    Network(std::vector<int> layer,std::vector<std::vector<double>> input,std::vector<double> output,double error);
    void forwardPropagation(int);  //forwardPropagation, receive the row of data
    void backwordPropagation(int);    //backwardPropagation
    double accumulateError();      //desired value – actual value
    void updateWeights(double);   //updateWeights
    std::vector<int> getLayers()const; //getLayers
    double getErrors()const;              //getErrors
    void setErrors(double);//set errors
    double getPredict(); //getYpredict
    std::vector<Matrix> getLayerOutputs()const;
    std::vector<std::vector<double>> getInputData()const;   //getInputData X
    std::vector<double> getOutputData()const;    //getOutputData Y
    std::string toString();
    void setWeights(std::vector<Matrix>);
    void setBias(std::vector<Matrix>);
    void clearLayerOutput();
    std::vector<Matrix> getWeights() const;
    std::vector<Matrix> getBias()const;
    ~Network();//destructor
private:
    std::vector<int> layers;
    std::vector<Matrix> weights;
    std::vector<Matrix> bias;
    double errors;
    std::vector<double> outputErrors;
    std::vector<std::vector<double>> inputData;
    std::vector<double> outputData;
    std::vector<Matrix> layerInputs;
    std::vector<Matrix> layerOutputs;
    std::vector<Matrix> errorVector;
};
#endif /* Network_h */
