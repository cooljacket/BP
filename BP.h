#ifndef BPNN_H
#define BPNN_H

#include "Matrix.h"
#include <vector>
#include <fstream>
using namespace std;


typedef vector<vector<double> > Data;


class BP {
private:
	int numOfLayers;
	vector<int> sizes;
	double eta;
	bool DEBUG, Data_loaded;
	vector<Matrix> activations, zs, weights, Biases, sigmas, BiasesDelta, WeightsDelta;
	Matrix target;
	Data test_x, test_y, validation_x, validation_y;

public:
	BP(const vector<int>& sizes, double eta=1.0, int numOfLayers=3, bool DEBUG=false);
	void Train_batch(const Data& trainData, const Data& target);
	void Train_stochastic(const Data& trainData, const Data& target, int epochs, int mini_batch_size);
	int Test(const Data& input, const Data& target);
	void saveModel();
	void loadModel();
	void RegisterData(const Data& test_x, const Data& test_y, const Data& validation_x, const Data& validation_y);

private:
	void forward();
	void train_one(const vector<double>& trainData, const vector<double>& target);
	int test_one(const vector<double>& input);
	void ClearDelta();
	void adjust(int size);
	int getResult(const vector<double>& output);
	int getResult();
	vector<int> getRandomOrder(int size);

	// double getError();
	// double getError(const vector<double>& target);
	// void saveMatrix(fstream& out, const Data& data);
};

#endif