#include "Matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>


Matrix::Matrix() {}


Matrix::Matrix(int row, int col) {
	data = Data(row, vector<double>(col));
}


Matrix::Matrix(const vector<double>& v, TYPE row_or_column) {
	if (row_or_column == ROW)
		data = Data(1, v);
	else if (row_or_column == COLUMN) {
		data = Data(v.size());
		for (int i = 0; i < v.size(); ++i)
			data[i].push_back(v[i]);
	} else {
		printf("Unknown type arguments: %d\n", row_or_column);
		exit(0);
	}
}


Matrix::Matrix(const Data& v) {
	data = Data(v);
}


// randomly initialize the matrix to be at range (-RANGE, RANGE)
Matrix Matrix::randMatrix(int row, int col, double RANGE) {
	srand(time(NULL));
	Matrix ans;
	ans.data = Data(row, vector<double>(col));

	int RAND_MOD = 9973;
	for (int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j)
			ans.data[i][j] = rand() % RAND_MOD * 2.0 / RAND_MOD * RANGE - RANGE;

	return ans;
}


void Matrix::display() {
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < data[i].size(); ++j)
			printf("%lf ", data[i][j]);
		printf("\n");
	}
	printf("\n");
}


void Matrix::clear() {
	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			data[i][j] = 0;
}


double Matrix::sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}


Matrix Matrix::sigmoid(const Matrix& m) {
	Matrix ans(m);
	for (int i = 0; i < m.data.size(); ++i)
		for (int j = 0; j < m.data[i].size(); ++j)
			ans.data[i][j] = sigmoid(m.data[i][j]);

	return ans;
}


Matrix Matrix::sigmoid_diff(const Matrix& m) {
	Matrix ans(sigmoid(m));
	for (int i = 0; i < ans.data.size(); ++i)
		for (int j = 0; j < ans.data[i].size(); ++j)
			ans.data[i][j] *= (1.0 - ans.data[i][j]);

	return ans;
}


Matrix Matrix::dot(double d) {
	Matrix ans(*this);

	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			ans.data[i][j] *= d;

	return ans;
}


Matrix Matrix::dot(const Matrix& b) {
	assert(cmpSize(b));

	Matrix ans(b);
	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			ans.data[i][j] = data[i][j] * b.data[i][j];

	return ans;
}


// compare the size of the two matrix
bool Matrix::cmpSize (const Matrix& b) {
	if (data.size() != b.data.size())
		return false;
	if (!data.empty() && data[0].size() != b.data[0].size())
		return false;
	return true;
}


Matrix Matrix::transpose() {
	assert(!data.empty());
	Matrix ans(data[0].size(), data.size());

	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			ans.data[j][i] = data[i][j];

	return ans;
}


Matrix Matrix::operator - (const Matrix& b) {
	Matrix ans(*this);
	ans -= b;
	return ans;
}


Matrix& Matrix::operator -= (const Matrix& b) {
	assert(cmpSize(b));

	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			data[i][j] -= b.data[i][j];

	return *this;
}


Matrix Matrix::operator + (const Matrix& b) {
	Matrix ans(data);
	ans += b;
	return ans;
}


Matrix& Matrix::operator += (const Matrix& b) {
	assert(cmpSize(b));

	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			data[i][j] += b.data[i][j];

	return *this;
}


Matrix Matrix::operator * (const Matrix& b) {
	assert(!data.empty() && !data[0].empty());
	assert(!b.data.empty() && !b.data[0].empty());

	int row = data.size(), col = b.data[0].size(), mid = b.data.size();
	Matrix ans(row, col);

	for (int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j)
			for (int k = 0; k < mid; ++k)
				ans.data[i][j] += data[i][k] * b.data[k][j];

	return ans;	
}


vector<double> Matrix::getData() {
	vector<double> ans;
	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			ans.push_back(data[i][j]);

	return ans;
}


void Matrix::save(fstream& out) {
	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			out << data[i][j] << ' ';
	out << endl;
}


void Matrix::load(fstream& in, int row, int col) {
	*this = Matrix(row, col);
	for (int i = 0; i < data.size(); ++i)
		for (int j = 0; j < data[i].size(); ++j)
			in >> data[i][j];
}