//
//  Matrix.h
//  OOP Final Project Neural Network
//
//  Created by Michael_hzy on 2019/12/9.
//  Copyright © 2019 Michael_hzy. All rights reserved.
//

#ifndef Matrix_h
#define Matrix_h
#include<vector>
#include<iostream>
class Matrix
{
public:
    Matrix();
    Matrix(int n);
    Matrix(int n, int m);
    Matrix(int n, int m, double v);
    ~Matrix();
    int getRow() const;
    int getColumn() const;
    std::vector<std::vector<double>> getData() const;
    Matrix hadamard(const Matrix&);
    Matrix getTranspose() const;
    Matrix getIdentity() const;
    Matrix operator+(const Matrix&);
    Matrix operator-(const Matrix&);
    Matrix operator*(const Matrix&);
    Matrix operator*(double);
    static void setData(std::vector<double>,Matrix&);
    static Matrix sigmod(Matrix);
    static Matrix sigmodDerivative(Matrix);
private:
    int row;
    int column;
    std::vector<std::vector<double>> data;
    void setDataOne(double dataIn);
    double getNij(const Matrix&, int, int );
    
friend std::ostream& operator<<(std::ostream&, const Matrix&);
friend std::istream& operator>>(std::istream&, Matrix&);
};
#endif /* Matrix_h */
