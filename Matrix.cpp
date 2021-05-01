//
//  Matrix.cpp
//  OOP Final Project Neural Network
//
//  Created by Michael_hzy on 2019/12/9.
//  Copyright © 2019 Michael_hzy. All rights reserved.
//

#include<iomanip>
#include"Matrix.h"
#include<cmath>
#include <vector>
using namespace std;

Matrix::Matrix():row(3), column(3)
{
    setDataOne(0);
}

Matrix::Matrix(int n) : row(n), column(n)
{
    setDataOne(0);
}

Matrix::Matrix(int r, int c) : row(r), column(c)
{
    setDataOne(0);
}

Matrix::Matrix(int r, int c, double d) : row(r), column(c)
{
    setDataOne(d);
}

void Matrix::setDataOne(double dataIn)
{
    for (int r = 0; r < row; ++r)
    {
        std::vector<double> matrixRow;
        for (int c = 0; c < column; ++c)
        {
            matrixRow.push_back(dataIn);
        }
        data.push_back(matrixRow);
        matrixRow.clear();
    }
}

int Matrix::getRow() const {return row;}

int Matrix::getColumn() const {return column;}

std::vector<std::vector<double>> Matrix::getData() const { return data; }

Matrix Matrix::getIdentity() const
{
    if (row == column)
    {
        Matrix temp(row, column);
        std::vector<std::vector<double>> tempData= data;
        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < column; ++j)
            {
                if (i == j) tempData[i][j] = 1;
                else tempData[i][j] = 0;
            }
        }
        temp.data = tempData;
        return temp;
    }
    else
    {
        std::cout << "It's not a square matrix\n";
        return *this;
    }
    
}

Matrix Matrix::hadamard(const Matrix& second)
{
    if (row == second.getRow() && column == second.getColumn())
    {
        Matrix temp(row, column);
        for (int r = 0; r < row; ++r)
        {
            for (int c = 0; c < column; ++c)
            {
                temp.data[r][c] = data[r][c] * second.data[r][c];
            }
        }
        return temp;
    }
    else
    {
        std::cout << row << " by " << column << "matrix can not hadamard multiply with " << second.column << " by " << second.column << " matrix" <<endl;
        return *this ;
    }
    
}

Matrix Matrix::getTranspose() const
{
    Matrix temp(column, row);
    std::vector<std::vector<double>> tempData;
    for (int i = 0; i < column; ++i)
    {
        std::vector<double> tempDataRow;
        for (int j = 0; j < row; ++j)
        {
            tempDataRow.push_back(data[j][i]);
        }
        tempData.push_back(tempDataRow);
        tempDataRow.clear();
    }
    temp.data = tempData;
    return temp;
}


Matrix Matrix::operator+(const Matrix& second)
{
    if (row == second.row && column == second.column)
    {
        Matrix temp(row, column);
        for (int r = 0; r < row; ++r)
        {
            for (int c = 0; c < column; ++c)
            {
                temp.data[r][c]=data[r][c]+second.data[r][c];
            }
        }
        return temp;
    }
    else
    {
        std::cout << row << " by " << column << "matrix can not add with " << second.column << " by " << second.column << " matrix"<< endl;
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix& second)
{
    if (row == second.row && column == second.column)
    {
        Matrix temp(row, column);
        for (int r = 0; r < row; ++r)
        {
            for (int c = 0; c < column; ++c)
            {
                temp.data[r][c] = data[r][c] - second.data[r][c];
            }
        }
        return temp;
    }
    else
    {
        std::cout << row << " by " << column << "matrix can not subtract with " << second.column << " by " << second.column << "matrix";
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& second)
{
    if (column == second.row)
    {
        Matrix temp(row, second.getColumn());
        for (int tempr = 0; tempr < row; ++tempr)
        {
            for (int tempc = 0; tempc < second.getColumn(); ++tempc)
            {
                temp.data[tempr][tempc] = getNij(second, tempr, tempc);
            }
        }
        return temp;
    }
    else
    {
        std::cout << row << " by " << column << "matrix can not multiply with " << second.column << " by " << second.column << "matrix";
    }
    return *this;
}

Matrix Matrix::operator*(double times)
{
    Matrix temp(row, column);
    for (int tempr = 0; tempr < row; ++tempr) {
        for (int tempc = 0; tempc < column; ++tempc)
        {
            temp.data[tempr][tempc] = data[tempr][tempc] * times;
        }
    }
    return temp;
}

double Matrix::getNij(const Matrix& second, int i, int j)
{
    double nij = 0;
    for (int Item = 0; Item < column; ++Item)
    {
        nij += data[i][Item] * second.data[Item][j];
    }
    return nij;
}

Matrix::~Matrix() {}

std::ostream& operator<<(std::ostream& output, const Matrix& matrix)
{
    output<< "this is a " << matrix.getRow() << " * " << matrix.getColumn() << " matrix\n";
    std::vector<std::vector<double>> data = matrix.getData();
    for (int i = 0; i < matrix.getRow(); ++i)
    {
        for (int j = 0; j < matrix.getColumn(); ++j)
        {
            output <<std::setw(8)<< data[i][j]<<"\t";
        }
        output << std::endl;
    }
    return output;
}

std::istream& operator>>(std::istream& input, Matrix& matrix)
{
    for (int i = 0; i < matrix.getRow(); ++i)
    {
        for (int j = 0; j < matrix.getColumn(); ++j)
        {
            input>>std::setw(1)>> matrix.data[i][j];
        }
    }
    return input;
}

void Matrix::setData(vector<double> temp, Matrix&matrix)
{
   for (int i = 0; i < matrix.getRow(); ++i)
   {
       for (int j = 0; j < matrix.getColumn(); ++j)
       {
           matrix.data[i][j]=temp[i * matrix.getColumn()+j];
       }
   }
}

Matrix Matrix::sigmod(Matrix tempMatrix)
{
    for (int i = 0; i < tempMatrix.getRow(); ++i)
    {
        for (int j = 0; j < tempMatrix.getColumn(); ++j)
        {
            tempMatrix.data[i][j] = 2 / (1 + exp(-tempMatrix.data[i][j])) - 1;
        }
    }
    return tempMatrix;
}

Matrix Matrix::sigmodDerivative(Matrix tempMatrix)
{
    for (int i = 0; i < tempMatrix.getRow(); ++i)
    {
        for (int j = 0; j < tempMatrix.getColumn(); ++j)
        {
            tempMatrix.data[i][j] = 0.5 * (2 / (1 + exp(-tempMatrix.data[i][j]))) * (2 - 2 / (1 + exp(-tempMatrix.data[i][j])));
        }
    }
    return tempMatrix;
}
