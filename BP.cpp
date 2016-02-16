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


BP::BP(int inputSize, int hiddenSize, int outputSize, double eta, bool DEBUG)
:inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), eta(eta), 
RAND_MOD(9973), epsilon_init(0.12), DEBUG(DEBUG)
{
	input = vector<double>(inputSize);
	hidden = vector<double>(hiddenSize);
	output = vector<double>(outputSize);
	target = vector<double>(outputSize);

	z_hid = vector<double>(hiddenSize);
	z_out = vector<double>(outputSize);

	hidBias = hidBiasDelta = vector<double>(hiddenSize);
	outBias = outBiasDelta = vector<double>(outputSize);

	w_hid_in = hidDelta = vector<vector<double> >(hiddenSize, vector<double>(inputSize));
	w_out_hid = outDelta = vector<vector<double> >(outputSize, vector<double>(hiddenSize));

	RandomizeWeights(w_hid_in);
	RandomizeWeights(w_out_hid);
	RandomizeBias(hidBias);
	RandomizeBias(outBias);
}


// randomly initialize the weights network
void BP::RandomizeWeights(vector<vector<double> >& w) {
	srand(time(NULL));

	for (int i = 0; i < w.size(); ++i)
		for (int j = 0; j < w[i].size(); ++j)
			w[i][j] = rand() % RAND_MOD * 1.0 / RAND_MOD * 2.0 * epsilon_init - epsilon_init;
}


// randomly initialize the bias
void BP::RandomizeBias(vector<double>& bias) {
	srand(time(NULL));

	for (int i = 0; i < bias.size(); ++i)
		bias[i] = rand() % RAND_MOD * 1.0 / RAND_MOD * 2.0 * epsilon_init - epsilon_init;
}


void BP::toZero(vector<double>& v) {
	for (int i = 0; i < v.size(); ++i)
		v[i] = 0;
}


// clear the delta matrix to be zeros
void BP::ClearDelta() {
	for (int i = 0; i < hidDelta.size(); ++i)
		toZero(hidDelta[i]);
	for (int i = 0; i < outDelta.size(); ++i)
		toZero(outDelta[i]);
	toZero(outBiasDelta);
	toZero(hidBiasDelta);
}


// train a single instance
void BP::train_one(const vector<double>& trainData, const vector<double>& target) {
	loadInput(trainData);
	loadTarget(target);
	forward();
	calculateDelta();
}


// test a single instance
// return: the predict result using the current neural network
int BP::test_one(const vector<double>& input) {
	loadInput(input);
	forward();
	return getResult();
}


// load the input vector
void BP::loadInput(const vector<double>& input) {
	assert(input.size() == this->input.size());
	copy(input.begin(), input.end(), this->input.begin());
}


// load the target vector
void BP::loadTarget(const vector<double>& target) {
	assert(target.size() == this->target.size());
	copy(target.begin(), target.end(), this->target.begin());
}


// forward propagation between the neighbor layers
void BP::forward(const vector<double>& Layer_in, vector<double>& Layer_out, vector<double>& z, const vector<vector<double> >& w, const vector<double>& bias) {
	for (int j = 0; j < Layer_out.size(); ++j) {
		double sum = 0;
		for (int i = 0; i < Layer_in.size(); ++i)
			sum += w[j][i] * Layer_in[i];
		z[j] = sum + bias[j];
		Layer_out[j] = sigmoid(z[j]);
	}
}


// forward propagation
void BP::forward() {
	forward(input, hidden, z_hid, w_hid_in, hidBias);
	forward(hidden, output, z_out, w_out_hid, outBias);
}


// calculate output layer error
vector<double> BP::outputErr() {
	vector<double> sigma(outDelta.size());
	vector<double> z_diff = sigmoid_diff(z_out);

	for (int i = 0; i < sigma.size(); ++i)
		sigma[i] = (output[i] - target[i]) * z_diff[i];
	return sigma;
}


// calculate hidden layer error
vector<double> BP::hiddenErr(const vector<double>& sigma_out) {
	vector<double> sigma(hidDelta.size());
	vector<double> z_diff = sigmoid_diff(z_hid);

	for (int j = 0; j < hidDelta.size(); ++j) {
		double sum = 0;
		for (int k = 0; k < outDelta.size(); ++k)
			sum += w_out_hid[k][j] * sigma_out[k];
		sigma[j] = sum * z_diff[j];
	}

	return sigma;
}


void BP::addWeightDelta(vector<vector<double> >& w_delta, const vector<double>& sigma, const vector<double>& Layer_in) {
	for (int j = 0; j < sigma.size(); ++j)
		for (int k = 0; k < Layer_in.size(); ++k)
			w_delta[j][k] += Layer_in[k] * sigma[j];
}


void BP::addBiasDelta(vector<double>& bias, const vector<double>& sigma) {
	for (int j = 0; j < sigma_out.size(); ++j)
		outBiasDelta[j] += sigma_out[j];
}


void BP::calculateDelta() {
	vector<double> sigma_out = outputErr();
	vector<double> sigma_hid = hiddenErr(sigma_out);

	addWeightDelta(outDelta, sigma_out, hidden);
	addWeightDelta(hidDelta, sigma_hid, input);
	addBiasDelta(outBiasDelta, sigma_out);
	addBiasDelta(hidBiasDelta, sigma_hid);
}


void BP::adjustWeight(vector<vector<double> >& delta, vector<vector<double> >& w, int size) {
	for (int i = 0; i < w.size(); ++i)
		for (int j = 0; j < w[i].size(); ++j)
			w[i][j] -= eta * delta[i][j] / size;
}


void BP::adjustBias(vector<double>& delta, vector<double>& bias, int size) {
	for (int i = 0; i < bias.size(); ++i)
		bias[i] -= eta * delta[i] / size;
}


void BP::adjust(int size) {
	adjustWeight(outDelta, w_out_hid, size);
	adjustWeight(hidDelta, w_hid_in, size);
	adjustBias(outBiasDelta, outBias, size);
	adjustBias(hidBiasDelta, hidBias, size);
}


double BP::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}


vector<double> BP::sigmoid_diff(const vector<double>& v) {
	vector<double> ans(v.size());
	for (int i = 0; i < v.size(); ++i) {
		double z = sigmoid(v[i]);
		ans[i] = z * (1.0 - z);
	}
	return ans;
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
	vector<double> out(output);
	return getResult(out);
}


// save the weights into file "model.txt"
void BP::saveModel() {
	fstream out("data/model.txt", ios::out);
	out << input.size() << ' ' << hidden.size() << ' ' << output.size() << ' ' << eta << endl;

	for (int i = 0; i < hidBias.size(); ++i)
		out << hidBias[i] << ' ';
	out << endl;

	for (int i = 0; i < outBias.size(); ++i)
		out << outBias[i] << ' ';
	out << endl;

	for (int j = 0; j < hidden.size(); ++j) {
		for (int i = 0; i < input.size(); ++i)
			out << w_hid_in[j][i] << ' ';
		out << endl;
	}

	for (int j = 0; j < output.size(); ++j) {
		for (int i = 0; i < hidden.size(); ++i)
			out << w_out_hid[j][i] << ' ';
		out << endl;
	}
	out.close();

	printf("The model has been saved...\n");
}


// load the weights from file "model.txt"
void BP::loadModel() {
	fstream in("data/model.txt", ios::in);
	int inputSize, hiddenSize, outputSize;
	in >> inputSize >> hiddenSize >> outputSize >> eta;
	*this = BP(inputSize, hiddenSize, outputSize, eta);

	for (int i = 0; i < hidBias.size(); ++i)
		in >> hidBias[i];

	for (int i = 0; i < outBias.size(); ++i)
		in >> outBias[i];

	for (int j = 0; j < hiddenSize; ++j) {
		for (int i = 0; i < inputSize; ++i)
			in >> w_hid_in[j][i];
	}

	for (int j = 0; j < outputSize; ++j) {
		for (int i = 0; i < hiddenSize; ++i)
			in >> w_out_hid[j][i];
	}
	in.close();

	printf("The model has been loaded...\n");
}


double BP::getError(const vector<double>& target) {
	assert(target.size() == output.size());

	vector<double> res(target.size());
	res[getResult()] = 1;
	double err = 0.0;

	for (int i = 0; i < target.size(); ++i)
		err += (target[i] - res[i])*(target[i] - res[i]);

	return err * 0.5;
}


vector<int> BP::getRandomOrder(int size) {
	vector<int> order(size);
	for (int i = 0; i < order.size(); ++i)
		order[i] = i;
	random_shuffle(order.begin(), order.end());
	return order;
}


// the data order should be random...
void BP::Train_online(const vector<vector<double> >& trainData, const vector<vector<double> >& target) {
	assert(trainData.size() == target.size());

	for (int i = 0; i < trainData.size(); ++i) {
		ClearDelta();
		train_one(trainData[i], target[i]);
		adjust(1);
	}
}


// TODO: train using all the input data to update the weights network
// return: the error of the current weights network
void BP::Train_batch(const vector<vector<double> >& trainData, const vector<vector<double> >& target) {
	assert(trainData.size() == target.size());
	ClearDelta();

	for (int i = 0; i < trainData.size(); ++i)
		train_one(trainData[i], target[i]);

	adjust(target.size());
}


void BP::Train_stochastic(const vector<vector<double> >& trainData, const vector<vector<double> >& target, int epochs, int mini_batch_size) {
	int total = trainData.size();
	vector<int> order(getRandomOrder(total));
	for (int e = 0; e < epochs; ++e) {
		clock_t start = clock();

		for (int i = 0; i < total; i += mini_batch_size) {
			// get the data for trainning
			int size = min(mini_batch_size, total-i);
			vector<vector<double> > data(size), y(size);
			for (int j = 0; j < size; ++j) {
				int index = order[i+j];
				data[j] = trainData[index];
				y[j] = target[index];
			}

			Train_batch(data, y);
		}

		int correct = Test(trainData, target);
		printf("Epochs [%d]:\t %d / %d (%lf%%)\n", e, correct, total, correct*100.0/total);
		if (DEBUG)
			printf("cost time=%lf, error=%lf%%\n", double((clock() - start) * 1.0 / CLOCKS_PER_SEC), correct*100.0/total);
	}
}


void BP::saveMatrix(fstream& out, const vector<vector<double> >& data) {
	out << setprecision(6) << fixed;
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < data[i].size(); ++j)
			out << data[i][j] << ' ';
		out << endl;
	}
	out << "\n\n";
}


// TODO: test all the input data using the well-trained network
// return: the correct rate of this model
int BP::Test(const vector<vector<double> >& input, const vector<vector<double> >& target) {
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