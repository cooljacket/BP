#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>
using namespace std;

enum TYPE
{
	ROW, COLUMN, MATRIX
};

typedef vector<vector<double> > Data;

class Matrix
{
private:
	Data data;

public:
	Matrix();
	Matrix(int row, int col);
	Matrix(const vector<double>& v, TYPE row_or_column=COLUMN);
	Matrix(const Data& v);

	void display();
	Matrix dot(const Matrix& b);
	Matrix dot(double d);
	bool cmpSize (const Matrix& b);
	Matrix transpose();
	void clear();
	vector<double> getData();

	Matrix operator - (const Matrix& b);
	Matrix& operator -= (const Matrix& b);
	Matrix operator + (const Matrix& b);
	Matrix& operator += (const Matrix& b);
	Matrix operator * (const Matrix& b);

	void save(fstream& out);
	void load(fstream& in, int row, int col);

	static Matrix randMatrix(int row, int col, double RANGE);
	static double sigmoid(double z);
	static Matrix sigmoid(const Matrix& m);
	static Matrix sigmoid_diff(const Matrix& m);
};

#endif