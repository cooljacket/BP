#ifndef BPNN_H
#define BPNN_H

#include <vector>
#include <fstream>
using namespace std;

class BP {
private:
	int RAND_MOD, inputSize, hiddenSize, outputSize;
	double epsilon_init, eta;
	bool DEBUG;

	vector<double> input, hidden, output, target, hidBias, outBias, z_hid, z_out, outBiasDelta, hidBiasDelta;
	vector<double> sigma_out, sigma_hid;
	vector<vector<double> > w_hid_in, w_out_hid, outDelta, hidDelta;

public:
	BP(int inputSize, int hiddenSize, int outputSize, double eta=1.0, bool DEBUG=true);
	void Train_batch(const vector<vector<double> >& trainData, const vector<vector<double> >& target);
	void Train_online(const vector<vector<double> >& trainData, const vector<vector<double> >& target);
	void Train_stochastic(const vector<vector<double> >& trainData, const vector<vector<double> >& target, int epochs, int mini_batch_size);
	int Test(const vector<vector<double> >& input, const vector<vector<double> >& target);
	void saveModel();
	void loadModel();

	static double sigmoid(double x);
	static vector<double> sigmoid_diff(const vector<double>& v);

private:	
	void RandomizeWeights(vector<vector<double> >& w);
	void RandomizeBias(vector<double>& bias);
	void toZero(vector<double>& v);
	void ClearDelta();
	void train_one(const vector<double>& trainData, const vector<double>& target);
	int test_one(const vector<double>& input);
	void loadInput(const vector<double>& input);
	void loadTarget(const vector<double>& target);
	void forward(const vector<double>& Layer_in, vector<double>& Layer_out, vector<double>& z, const vector<vector<double> >& w, const vector<double>& bias);
	void forward();
	vector<double> outputErr();
	vector<double> hiddenErr(const vector<double>& sigma_out);
	void addWeightDelta(vector<vector<double> >& w_delta, const vector<double>& sigma, const vector<double>& Layer_in);
	void addBiasDelta(vector<double>& bias, const vector<double>& sigma);
	void calculateDelta();
	void adjustWeight(vector<vector<double> >& delta, vector<vector<double> >& w, int size);
	void adjustBias(vector<double>& delta, vector<double>& bias, int size);
	void adjust(int size);
	int getResult(const vector<double>& output);
	int getResult();
	double getError();
	double getError(const vector<double>& target);
	vector<int> getRandomOrder(int size);
	void saveMatrix(fstream& out, const vector<vector<double> >& data);
};

#endif