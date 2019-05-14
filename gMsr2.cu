//============================================================================
// Name        : gMsr2.cu
// Author      : Aurelio Lopez-Fernandez
// Version     :
// Copyright   : Your copyright notice
// Description : Esta version calcula todas las medias y media total del bicluster en cada elemento del bicluster.
//============================================================================

#include <thread>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>
#include <string>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <math.h>
#include <mutex>

using namespace std;

//introduceParameters()
string biclustersFile, matrixFile, outputFile;
double delta;
unsigned int deviceCount;

//Matrix biclusters
ulong cRowsmBiclusters, cColsmBiclusters, *mBiclusters;
__constant__ ulong rowsmBiclusters;
__constant__ ulong colsmBiclusters;

//Matrix expression gene
ulong cRowsmExpression, cColsmExpression;
__constant__ ulong rowsmExpression;
__constant__ ulong colsmExpression;
double *mExpression;

// GPU variables
ulong maxThreadsPerBlock, maxBlocksPerGrid, maxIteratorGPU, lastBlocksGrid;

// Delta biclusters
struct compMsr {
    bool operator() (const pair<ulong, double>& elem1, const pair<ulong, double>& elem2) const {
    	return elem1.second < elem2.second;
    }
};
set<pair<ulong, double>, compMsr> setMsr;

__global__ void generateMsr(int idGpu, ulong min, double *mGeneExpression, ulong *gBiclusters, double *mResultMsr, ulong totalBiclusters, int idKernel, ulong totalFor) {
	ulong uBicluster = blockIdx.x * blockDim.x + threadIdx.x + (totalFor*(idKernel-1));

	if(uBicluster < totalBiclusters){
		ulong iRows = *(gBiclusters + uBicluster * colsmBiclusters + 0);
		ulong iCols = *(gBiclusters + uBicluster * colsmBiclusters + 1);

		double fMsr = 0;
		for(ulong contRow=0; contRow < iRows ; contRow++){
			ulong iRow = *(gBiclusters + uBicluster * colsmBiclusters + (2+contRow));
			for(ulong contCols=0; contCols < iCols; contCols++){
				ulong iCol = *(gBiclusters + uBicluster * colsmBiclusters + (2+iRows+contCols));
				double element_aij = *(mGeneExpression + iRow * colsmExpression + iCol); // Element aij

				// 1) Get total bicluster media and row bicluster media
				double mediaBicluster = 0, mediaRows = 0;
				for(ulong contRowTotalBicMedia=0; contRowTotalBicMedia < iRows ; contRowTotalBicMedia++){
					ulong iRowTotalBicMedia = *(gBiclusters + uBicluster * colsmBiclusters + (2+contRowTotalBicMedia));
					for(ulong contColsTotalBicMedia=0; contColsTotalBicMedia < iCols; contColsTotalBicMedia++){
						ulong iColTotalBicMedia = *(gBiclusters + uBicluster * colsmBiclusters + (2+iRows+contColsTotalBicMedia));
						if(contRowTotalBicMedia == contRow){
							mediaRows += *(mGeneExpression + iRow * colsmExpression + iColTotalBicMedia);
						}
						mediaBicluster += *(mGeneExpression + iRowTotalBicMedia * colsmExpression + iColTotalBicMedia);
					}
				}
				mediaRows /= iCols;
				mediaBicluster /= (iRows*iCols);

				// 2) Get col bicluster media
				double mediaCols = 0;
				for(ulong contColTotalBicMedia=0; contColTotalBicMedia < iRows; contColTotalBicMedia++){
					ulong iRowBicMedia = *(gBiclusters + uBicluster * colsmBiclusters + (2+contColTotalBicMedia));
					mediaCols += *(mGeneExpression + iRowBicMedia * colsmExpression + iCol);
				}
				mediaCols /= iRows;

				// 3) Calculate MSR
				fMsr += pow(element_aij - mediaRows - mediaCols + mediaBicluster,2);
			}
		}
		fMsr /= (iRows*iCols);
		*(mResultMsr + uBicluster) = fMsr;
	}
}

void introduceParameters() {

	// PARAMETER 1: Biclusters file
	biclustersFile = "/home/principalpc/G-MSR/Results/prueba.csv";

	// PARAMETER 2: Matrix file
	matrixFile = "/home/principalpc/G-MSR/Matrix/yeast.matrix";

	//PARAMETER 3: OUTPUT
	delta = 2000;

	//PARAMETER 4: GPus number
	deviceCount = 2;

	//PARAMETER 5: Output file
	outputFile = "/home/principalpc/G-MSR/output.csv";

}

void biclustersReader() {
	string line;
	ifstream myfile(biclustersFile.c_str());
	// 1) Read file
	if (myfile.is_open()) {
		vector< vector<ulong> > aRows, aCols;
		ulong maxRows = 0, maxCols = 0;

		while (getline(myfile, line)) {
			vector<ulong> rows, cols;
			string sRows = line.substr(0, line.find(";", 0));
			string sCols = line.substr(line.find(";", 0) + 1, line.size());

			//1.1) Prepare bicluster rows
			stringstream sr(sRows);
			ulong i;
			while (sr >> i) {
				rows.push_back(i);
				if (sr.peek() == ',') {
					sr.ignore();
				}
			}

			//1.2) Prepare bicluster cols
			stringstream sc(sCols);
			while (sc >> i) {
				cols.push_back(i);
				if (sc.peek() == ',') {
					sc.ignore();
				}
			}

			// 1.3) Calculate dimension of matrix
			if(rows.size() > maxRows){
				maxRows = rows.size();
			}

			if(cols.size() > maxCols){
				maxCols = cols.size();
			}

			aRows.push_back(rows);
			aCols.push_back(cols);
			cRowsmBiclusters++;
		}
		myfile.close();

		// 2) Create matrix
		cColsmBiclusters = 2+maxRows+maxCols;
		mBiclusters = (ulong *) malloc(cRowsmBiclusters * cColsmBiclusters * sizeof(ulong));

		// 3) Fill the matrix
		for(ulong lBic = 0; lBic < cRowsmBiclusters; lBic++){
			*(mBiclusters + lBic * cColsmBiclusters + 0) = aRows[lBic].size();
			*(mBiclusters + lBic * cColsmBiclusters + 1) = aCols[lBic].size();
			for(ulong lRows = 0; lRows < aRows[lBic].size() ; lRows++){
				*(mBiclusters + lBic * cColsmBiclusters + (2+lRows)) = (aRows[lBic])[lRows];
			}
			for(ulong lCols = 0; lCols < aCols[lBic].size() ; lCols++){
				*(mBiclusters + lBic * cColsmBiclusters + (2+aRows[lBic].size()+lCols)) = (aCols[lBic])[lCols];
			}
			for(ulong lRest = 2+aRows[lBic].size()+aCols[lBic].size(); lRest < cColsmBiclusters ; lRest++){
				*(mBiclusters + lBic * cColsmBiclusters + lRest) = 0;
			}
		}
	} else {
		cout << "Unable to open file " << endl;
	}
}

void readerMatrix() {

	// 1) Prepare Matrix from file
	cRowsmExpression = 0;
	cColsmExpression = 0;
	vector<string> rowsArray_Aux;
	string line;
	ifstream myfile(matrixFile.c_str());
	if (myfile.is_open()) {
		// 2.1) Get number of rows
		while (getline(myfile, line)) {
			rowsArray_Aux.push_back(line);
			// 2.2) Get number of columns
			if (cRowsmExpression == 0) {
				for (int k = line.size() - 1; k >= 0; k--) {
					if (line[k] == ',') {
						cColsmExpression++;
					}
				}
				cColsmExpression++;
			}
			cRowsmExpression++;
		}
		myfile.close();

		// 2) Create matrix
		mExpression = (double *) malloc(cRowsmExpression * cColsmExpression * sizeof(double));

		// 3) Fill matrix
		for (ulong j = 0; j < cRowsmExpression; j++) {
			string row = rowsArray_Aux[j];
			stringstream ss(row);
			double i;
			int contCols = 0;
			while (ss >> i) {
				*(mExpression + j * cColsmExpression + contCols) = i;
				if (ss.peek() == ',') {
					ss.ignore();
				}
				contCols++;
			}
		}
	} else {
		cout << "Unable to open file " << endl;
	}
}

void saveFile(){
	ofstream myfile(outputFile);
	if (myfile.is_open()) {
		myfile << "BICLUSTER_ID,MSR\n";
		set<pair<ulong, double>, compMsr>::iterator itr;
		for (itr = setMsr.begin(); itr != setMsr.end(); itr++) {
			pair<ulong, double> x = *itr;
			myfile << x.first << "," << x.second << "\n";
		}
		myfile.close();
	} else cout << "Unable to save file";
}

ulong *splitBiclusterMatrix(ulong biclustersPerChunk, ulong min){
	ulong *cBiclusters = (ulong *) malloc(biclustersPerChunk * cColsmBiclusters * sizeof(ulong));
	for(int r = 0; r < biclustersPerChunk; r++){
		for(int c = 0; c < cColsmBiclusters; c++){
			*(cBiclusters + r * cColsmBiclusters + c) = *(mBiclusters + (r+min) * cColsmBiclusters + c);
		}
	}
	return cBiclusters;
}

void prepareGpu1D(ulong lNumber){
	int device;
	cudaGetDevice(&device);
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	lastBlocksGrid = 1;
	maxIteratorGPU = 0;
	maxThreadsPerBlock = lNumber; // Case 1: 0 < lNumber <= prop.maxThreadsPerBlock
	if(lNumber > prop.maxThreadsPerBlock){ // Case 2: lNumber > prop.maxThreadsPerBlock && Supported GPU in a for
		maxThreadsPerBlock = prop.maxThreadsPerBlock;
		maxBlocksPerGrid = lNumber / prop.maxThreadsPerBlock;
		lastBlocksGrid = lNumber / prop.maxThreadsPerBlock;
		if(lNumber % prop.maxThreadsPerBlock != 0){
			maxBlocksPerGrid++;
			lastBlocksGrid++;
		}
		if(maxBlocksPerGrid > prop.maxGridSize[1]){ // Case 3: Not supported GPU with a for --> Split patterns in multiple for
			maxIteratorGPU = maxBlocksPerGrid / prop.maxGridSize[1];
			lastBlocksGrid = maxBlocksPerGrid - (maxIteratorGPU * prop.maxGridSize[1]);
			maxBlocksPerGrid = prop.maxGridSize[1];
		}
	}
}

void threadsPerDevice(int idGpu, cudaStream_t s, ulong *chunks, ulong *biclustersPerChunk, double *mGeneExpression, ulong totalBiclustersPerGpu, mutex *m) {
	cudaSetDevice(idGpu);

	// 1) The minimum and maximum index of each GPU is calculated taking into account the chunks.
	ulong minIndexBicByGpu = 0;
	ulong maxIndexBicByGpu = chunks[0]*biclustersPerChunk[0];
	for(int i=1; i<=idGpu; i++){ // Multi-GPU
		minIndexBicByGpu += chunks[i-1]*biclustersPerChunk[i-1];
		maxIndexBicByGpu += chunks[i]*biclustersPerChunk[i];
	}
	maxIndexBicByGpu -= 1;

	for(ulong largeScale = 0; largeScale < chunks[idGpu] ; largeScale++){
		ulong min = minIndexBicByGpu+(largeScale*biclustersPerChunk[idGpu]);
		ulong max = (minIndexBicByGpu+(largeScale*biclustersPerChunk[idGpu])+biclustersPerChunk[idGpu])-1;

		// 2) Result Array (MSR) to GPU global memory & CPU
		double *gResultMsr, *cResultMsr;
		cResultMsr = (double *) malloc(biclustersPerChunk[idGpu] * sizeof(double));
		cudaMalloc((void **)&gResultMsr, biclustersPerChunk[idGpu] * sizeof(double));
		cudaMemset(gResultMsr, 0, biclustersPerChunk[idGpu] * sizeof(double));

		// 3) Dynamic split of the Bicluster Matrix file to GPU devices.
		ulong *gBiclusters;
		ulong *cBiclusters = splitBiclusterMatrix(biclustersPerChunk[idGpu], min);
		cudaMallocHost((void**) &gBiclusters, biclustersPerChunk[idGpu] * cColsmBiclusters * sizeof(ulong));
		cudaMemcpy(gBiclusters, cBiclusters, biclustersPerChunk[idGpu] * cColsmBiclusters * sizeof(ulong), cudaMemcpyHostToDevice);

		// 4) Calculate MSR by GPU devices
		prepareGpu1D(biclustersPerChunk[idGpu]);
		for(int i=1; i <= maxIteratorGPU; i++){
			generateMsr<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(idGpu, min, mGeneExpression, gBiclusters, gResultMsr, biclustersPerChunk[idGpu], i, maxBlocksPerGrid);
		}
		generateMsr<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(idGpu, min, mGeneExpression, gBiclusters, gResultMsr, biclustersPerChunk[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

		// 5) Save results - Ordered and filtered delta biclusters
		cudaMemcpy(cResultMsr, gResultMsr, biclustersPerChunk[idGpu] * sizeof(double), cudaMemcpyDeviceToHost);
		for(ulong r=0; r < biclustersPerChunk[idGpu]; r++){
			double fMsr = *(cResultMsr + r);
			if(fMsr >= 0 && fMsr <= delta){
				cout << r+minIndexBicByGpu << endl;
				pair<ulong, double> x = make_pair(r+minIndexBicByGpu+1,fMsr);
				m->lock();
				setMsr.insert(x);
				m->unlock();
			}
		}

		// 6) Free memory spaces.
		cudaFree(gResultMsr);
		cudaFree(gBiclusters);
		free(cResultMsr);
		free(cBiclusters);
	}
	cudaFree(mExpression);
}

void runAlgorithm() {

	// Prepare GPU large-scale data
	// Chunks: Number of a GPU runs due to the input arrays are larger than the available memory of the GPU.
	// biclustersPerChunk: How many biclusters per chunk can process a GPU device.

	cudaStream_t s[deviceCount];
	thread threads[deviceCount];
	ulong *chunks = (ulong *) malloc(deviceCount * sizeof(ulong));
	ulong *biclustersPerChunk = (ulong *) malloc(deviceCount * sizeof(ulong));
	ulong totalBiclustersPerGpu = cRowsmBiclusters/deviceCount;

	for(int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		cudaStreamCreate(&s[i]);
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		double availableMemory = ((3*prop.totalGlobalMem)/4 - (cRowsmExpression * cColsmExpression * sizeof(double))); // Gene expression matrix (mExpression)
		double sizeResult = (totalBiclustersPerGpu * sizeof(double)); //Result Matrix (MSR)
		double sizeBiclusters = (totalBiclustersPerGpu * cColsmBiclusters * sizeof(ulong)); // Bicluster Matrix File
		chunks[i] = ((sizeResult+sizeBiclusters)/availableMemory)+1;
		biclustersPerChunk[i] = totalBiclustersPerGpu / chunks[i];
		if(deviceCount > 1 && totalBiclustersPerGpu%deviceCount!=0 && i==deviceCount-1){ // If it is the last GPU and there are still more biclusters to execute, the last GPU execute the biclusters exceded.
			biclustersPerChunk[i] = (totalBiclustersPerGpu+(totalBiclustersPerGpu%deviceCount)) / chunks[i];
		}

		// Copy constant variables
		cudaMemcpyToSymbol(*(&colsmBiclusters),&cColsmBiclusters,sizeof(ulong),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(*(&rowsmBiclusters),&cRowsmBiclusters,sizeof(ulong),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(*(&rowsmExpression),&cRowsmExpression,sizeof(ulong),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(*(&colsmExpression),&cColsmExpression,sizeof(ulong),0,cudaMemcpyHostToDevice);
	}

	// Prepare multi-GPU threads
	mutex m;
	for(int i=0; i<deviceCount; i++){
		cudaSetDevice(i);
		double *mGeneExpression;
		cudaMallocHost((void**) &mGeneExpression, cRowsmExpression * cColsmExpression * sizeof(double));
		cudaMemcpy(mGeneExpression, mExpression, cRowsmExpression * cColsmExpression * sizeof(double), cudaMemcpyHostToDevice);
		threads[i] = thread(threadsPerDevice, i, s[i], chunks, biclustersPerChunk, mGeneExpression, totalBiclustersPerGpu, &m);
	}

	for(auto& th: threads){
		th.join();
	}
}

int main() {

	introduceParameters();
	readerMatrix();
	biclustersReader();
	runAlgorithm();

	// Print DELTA BICLUSTERS
	cout << "###############" << endl;
	cout << "G-MSR INFO:" << endl;
	cout << "###############" << endl;
	cout << "GPU Devices: " << deviceCount << endl;
	cout << "Delta filter: " << delta << endl;
	cout << "Results save in: " << outputFile << endl;
	cout << endl << endl;
	saveFile();

	return cudaDeviceReset();
}
