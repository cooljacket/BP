#include "BP.h"
#include <vector>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
using namespace std;


BP::BP(const vector<int>& sizes, double eta, int numOfLayers, bool DEBUG)
:eta(eta), DEBUG(DEBUG), Data_loaded(false), numOfLayers(numOfLayers), sizes(sizes),
activations(numOfLayers), zs(numOfLayers), weights(numOfLayers), Biases(numOfLayers), 
sigmas(numOfLayers), BiasesDelta(numOfLayers), WeightsDelta(numOfLayers)
{
	target = Matrix(sizes[sizes.size()-1], 1);

	activations[0] = Matrix(sizes[0], 1);
	for (int i = 1; i < sizes.size(); ++i) {
		activations[i] = Matrix(sizes[i], 1);
		zs[i] = Matrix(sizes[i], 1);
		sigmas[i] = Matrix(sizes[i], 1);	

		weights[i] = Matrix::randMatrix(sizes[i], sizes[i-1], 0.12);
		WeightsDelta[i] = Matrix(sizes[i], sizes[i-1]);

		Biases[i] = Matrix::randMatrix(sizes[i], 1, 1.0);
		BiasesDelta[i] = Matrix(sizes[i], 1);
	}
}


// train a single instance
void BP::train_one(const vector<double>& trainData, const vector<double>& target) {
	// load input and target
	activations[0] = Matrix(trainData);
	this->target = Matrix(target);

	// forward
	forward();

	// backpropagation
	for (int L = numOfLayers-1; L > 0; --L) {
		if (L == numOfLayers-1)
			sigmas[L] = (activations[L] - target).dot(Matrix::sigmoid_diff(zs[L]));
		else
			sigmas[L] = (weights[L+1].transpose() * sigmas[L+1]).dot(Matrix::sigmoid_diff(zs[L]));
		WeightsDelta[L] += sigmas[L] * activations[L-1].transpose();
		BiasesDelta[L] += sigmas[L];
	}
}


// test a single instance
// return: the predict result using the current neural network
int BP::test_one(const vector<double>& input) {
	activations[0] = Matrix(input, COLUMN);
	forward();
	return getResult();
}


void BP::forward() {
	for (int i = 1; i < numOfLayers; ++i) {
		zs[i] = weights[i]*activations[i-1] + Biases[i];
		activations[i] = Matrix::sigmoid(zs[i]);
	}
}


int BP::getResult(const vector<double>& output) {
	double maxN = output[0];
	int ans = 0;
	for (int i = 0; i < output.size(); ++i)
		if (output[i] > maxN) {
			maxN = output[i];
			ans = i;
		}

	return ans;
}


int BP::getResult() {
	return getResult(activations[numOfLayers-1].getData());
}


vector<int> BP::getRandomOrder(int size) {
	vector<int> order(size);
	for (int i = 0; i < order.size(); ++i)
		order[i] = i;
	random_shuffle(order.begin(), order.end());
	return order;
}


void BP::ClearDelta() {
	for (int i = 0; i < BiasesDelta.size(); ++i)
		BiasesDelta[i].clear();
	for (int i = 0; i < WeightsDelta.size(); ++i)
		WeightsDelta[i].clear();
}


void BP::adjust(int size) {
	for (int i = 0; i < weights.size(); ++i)
		weights[i] -= WeightsDelta[i].dot(eta / size);

	for (int i = 0; i < Biases.size(); ++i)
		Biases[i] -= BiasesDelta[i].dot(eta / size);
}


// TODO: train using all the input data to update the weights network
// return: the error of the current weights network
void BP::Train_batch(const Data& trainData, const Data& target) {
	assert(trainData.size() == target.size());
	ClearDelta();

	for (int i = 0; i < trainData.size(); ++i)
		train_one(trainData[i], target[i]);

	adjust(target.size());
}


void BP::Train_stochastic(const Data& trainData, const Data& target, int epochs, int mini_batch_size) {
	int total = trainData.size(), xsize = trainData[0].size(), ysize = target[0].size();
	vector<int> order(getRandomOrder(total));
	for (int e = 1; e <= epochs; ++e) {
		clock_t start = clock();

		for (int i = 0; i < total; i += mini_batch_size) {
			// get the data for trainning
			int size = min(mini_batch_size, total-i);
			Data data(size, vector<double>(xsize)), y(size, vector<double>(ysize));
			for (int j = 0; j < size; ++j) {
				int index = order[i+j];
				data[j] = trainData[index];
				y[j] = target[index];
			}

			Train_batch(data, y);
		}

		if (Data_loaded) {
			int fit = Test(trainData, target);
			int test = Test(test_x, test_y);
			int validation = Test(validation_x, validation_y);
			printf("Epochs [%d]:\tfit: %.3lf%%\ttest: %.3lf%%\tvalidation: %.3lf%%\n", 
				e, fit*100.0/total, test*100.0/test_x.size(), validation*100.0/validation_x.size());
		}
		if (DEBUG)
			printf("cost time=%lf\n", double((clock() - start) * 1.0 / CLOCKS_PER_SEC));
	}
}


// TODO: test all the input data using the well-trained network
// return: the correct rate of this model
int BP::Test(const Data& input, const Data& target) {
	int correct = 0;
	for (int i = 0; i < input.size(); ++i) {
		int predict_res = test_one(input[i]);
		int actual_res = getResult(target[i]);
		if (predict_res == actual_res)
			++correct;
		if (DEBUG)
			printf("predict_res=%d, actual_res=%d\n", predict_res, actual_res);
	}

	return correct;
}


void BP::RegisterData(const Data& test_x, const Data& test_y, const Data& validation_x, const Data& validation_y) {
	this->test_x = test_x;
	this->test_y = test_y;
	this->validation_x = validation_x;
	this->validation_y = validation_y;
	Data_loaded = true;
}



// save the weights into file "model.txt"
void BP::saveModel() {
	fstream out("data/model.txt", ios::out);
	out << numOfLayers << ' ' << eta << ' ';
	for (int i = 0; i < sizes.size(); ++i)
		out << sizes[i] << ' ';
	out << endl;

	for (int i = 1; i < Biases.size(); ++i)
		Biases[i].save(out);

	for (int i = 1; i < weights.size(); ++i)
		weights[i].save(out);

	out.close();

	printf("The model has been saved...\n");
}


// load the weights from file "model.txt"
void BP::loadModel() {
	fstream in("data/model.txt", ios::in);
	in >> numOfLayers >> eta;
	*this = BP(sizes, eta, numOfLayers, DEBUG);

	for (int i = 1; i < Biases.size(); ++i)
		Biases[i].load(in, sizes[i], 1);

	for (int i = 1; i < weights.size(); ++i)
		weights[i].load(in, sizes[i], sizes[i-1]);
	
	in.close();

	printf("The model has been loaded...\n");
}