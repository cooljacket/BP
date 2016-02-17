#include "BP.h"
#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <time.h>
using namespace std;

const int xsize = 28 * 28, ysize = 10, hidden_size = 30, numOfLayers = 3;
int train_data_num = 50000, test_data_num = 10000, vali_data_num = 10000, epochs = 30, mini_batch_size = 10;
double eta = 1.0;
bool DEBUG = false;

void dealCMDArgs(int argc, char* argv[]);
void CReadData(const char* fileName, vector<vector<double> >& x, vector<vector<double> >& y, int size, const char* tips);


int main(int argc, char** argv) {
	dealCMDArgs(argc, argv);

	clock_t start = clock();
	vector<vector<double> > train_x, test_x, train_y, test_y, vali_x, vali_y;
	CReadData("data/train_data.txt", train_x, train_y, train_data_num, "Reading training data:");
	CReadData("data/validation_data.txt", vali_x, vali_y, vali_data_num, "Reading validation data:");
	CReadData("data/test_data.txt", test_x, test_y, test_data_num, "Reading testing data:");
	printf("Read data cost time: %lf\n", double((clock() - start) * 1.0 / CLOCKS_PER_SEC));

	vector<int> sizes(numOfLayers);
	sizes[0] = xsize;
	sizes[1] = hidden_size;
	sizes[2] = ysize;
	BP bp(sizes, eta, numOfLayers, DEBUG);
	bp.RegisterData(test_x, test_y, vali_x, vali_y);
	bp.Train_stochastic(train_x, train_y, epochs, mini_batch_size);
	bp.saveModel();

	return 0;
}


void dealCMDArgs(int argc, char* argv[]) {
	for (int i = 1; i < argc; i+=2) {
		assert(argc >= i+1);
		char ch = argv[i][1];
		stringstream ss;
		ss << argv[i+1];
		switch (ch) {
			case 'e':
				ss >> eta;
				break;
			case 'r':
				ss >> train_data_num;
				break;
			case 't':
				ss >> test_data_num;
				break;
			case 'v':
				ss >> vali_data_num;
				break;
			case 'l':
				ss >> epochs;
				break;
			case 'd':
				ss >> DEBUG;
				break;
			case 'm':
				ss >> mini_batch_size;
				break;
			default:
				printf("Unknown arguments%s\n", argv[i]);
				exit(0);
				break;
		}
	}
}


void CReadData(const char* fileName, vector<vector<double> >& x, vector<vector<double> >& y, int size, const char* tips) {
	FILE* fid = fopen(fileName, "r");
	if (fid == NULL) {
		printf("Failed to open file: %s\n", fileName);
		exit(-1);
	}
	int label, num_of_instance = 0;
	double tmp;

	x = vector<vector<double> >(size, vector<double>(xsize));
	y = vector<vector<double> >(size, vector<double>(ysize));

	printf("%s\t", tips);
	while (fscanf(fid, "%d", &label)) {
		y[num_of_instance][label] = 1;
		for (int i = 0; i < xsize; ++i) {
			fscanf(fid, "%lf", &tmp);
			x[num_of_instance][i] = tmp;
		}
		
		if (++num_of_instance >= size)
			break;
	}
	printf("%d cases\n", num_of_instance);
	fclose(fid);
}