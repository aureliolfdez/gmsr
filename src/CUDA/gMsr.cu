//============================================================================
// Name        : gMsr_v2_1.cu
// Author      : Aurelio Lopez-Fernandez
// Version     :
// Copyright   : Your copyright notice
// Description : Esta version calcula primeramente las medias y las almacena en vectores. Despues simplemente calcula el MSR en funcion de las medias almacenadas.
//				 A nivel de GPU: Un bloque se encarga de procesar un bicluster y un hilo se encarga de ejecutar uno o un conjunto de tareas.
//============================================================================

#include <thread>
#include <list>
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
#include <math.h>
#include <mutex>
#include <chrono>
#include <tuple>
#include <sys/sysinfo.h>
#include <limits.h>
using namespace std;

//introduceParameters
string biclustersFile, matrixFile, outputFile;
double delta;
unsigned int deviceCount, biclustersOutput;

//Matrix biclusters
vector< vector<ulong> > aRows, aCols;
ulong cRowsmBiclusters, cColsmBiclusters, *mBiclusters;
__constant__ ulong rowsmBiclusters;
__constant__ ulong colsmBiclusters;

//Matrix expression gene
ulong cRowsmExpression, cColsmExpression;
__constant__ ulong rowsmExpression;
__constant__ ulong colsmExpression;
double *mExpression;

// GPU variables
ulong maxThreadsPerBlock, maxBlocksPerGrid, maxIteratorGPU, lastBlocksGrid, loopsByThread;

// Delta biclusters
struct Msr
{
	ulong uBicluster;
	double dMsr;

	Msr(ulong msrBicluster, double msrValue): uBicluster(msrBicluster), dMsr(msrValue)
	{
	}
	bool operator <(const Msr & msrObj) const
	{
		return dMsr < msrObj.dMsr;
	}
};
list<Msr> listMsr;

__global__ void calculateRowsTotalMedia(ulong loopByThread, int idGpu, double *mGeneExpression, ulong *gBiclusters, double *gMediaBicluster, double *gRowsMedia, ulong rowsgRowsMedia, ulong totalBiclusters, int idKernel, ulong totalFor) {
	ulong uBicluster = blockIdx.x + (totalFor*(idKernel-1));
	ulong uThread = threadIdx.x;
	if(uBicluster < totalBiclusters){
		__shared__ ulong iRows, iCols;
		__shared__ double mediaTotal;
		if(uThread == 0){
			mediaTotal = 0;
			iRows = *(gBiclusters + uBicluster * colsmBiclusters + 0);
			iCols = *(gBiclusters + uBicluster * colsmBiclusters + 1);
		}
		__syncthreads();
		double mediaSubTotal = 0;
		for(ulong contLoop=0; contLoop < loopByThread && (uThread*loopByThread)+contLoop < iRows ; contLoop++){
			ulong iRow = *(gBiclusters + uBicluster * colsmBiclusters + (2+(uThread*loopByThread)+contLoop));
			double media = 0;
			for(ulong contCols=0; contCols < iCols; contCols++){
				ulong iCol = *(gBiclusters + uBicluster * colsmBiclusters + (2+iRows+contCols));
				media += *(mGeneExpression + (iRow-1) * colsmExpression + (iCol-1));
			}
			mediaSubTotal += media;
			media /= iCols;
			*(gRowsMedia + uBicluster * rowsgRowsMedia + ((uThread*loopByThread)+contLoop)) = media;
		}
		atomicAdd(&mediaTotal, mediaSubTotal);
		__syncthreads();
		if(uThread == 0){
			mediaTotal /= (iRows*iCols);
			*(gMediaBicluster + uBicluster) = mediaTotal;
		}
	}
}

__global__ void calculateColsMedia(ulong loopByThread, int idGpu, ulong min, double *mGeneExpression, ulong *gBiclusters, double *gColsMedia, ulong colsgColsMedia, ulong totalBiclusters, int idKernel, ulong totalFor) {
	ulong uBicluster = blockIdx.x + (totalFor*(idKernel-1));
	ulong uThread = threadIdx.x;
	if(uBicluster < totalBiclusters){
		__shared__ ulong iRows, iCols;
		if(uThread == 0){
			iRows = *(gBiclusters + uBicluster * colsmBiclusters + 0);
			iCols = *(gBiclusters + uBicluster * colsmBiclusters + 1);
		}
		__syncthreads();
		for(ulong contLoop=0; contLoop < loopByThread && (uThread*loopByThread)+contLoop < iCols; contLoop++){
			ulong iCol = *(gBiclusters + uBicluster * colsmBiclusters + (2+iRows+(uThread*loopByThread)+contLoop));
			double media = 0;
			for(ulong contRow=0; contRow < iRows; contRow++){
				ulong iRow = *(gBiclusters + uBicluster * colsmBiclusters + (2+contRow));
				media += *(mGeneExpression + (iRow-1) * colsmExpression + (iCol-1));
			}
			media /= iRows;
			*(gColsMedia + uBicluster * colsgColsMedia + ((uThread*loopByThread)+contLoop)) = media;
		}
	}
}

__global__ void calculateMsr(ulong loopByThread, int idGpu, ulong min, double *gResultMsr, double *mGeneExpression, ulong *gBiclusters, double *gMediaBicluster, double *gRowsMedia, double *gColsMedia, ulong rowsgRowsMedia, ulong colsgColsMedia, ulong totalBiclusters, int idKernel, ulong totalFor) {
	ulong uBicluster = blockIdx.x + (totalFor*(idKernel-1));
	ulong uThread = threadIdx.x;
	if(uBicluster < totalBiclusters){
		__shared__ ulong iRows, iCols;
		__shared__ double fMsr;
		if(uThread == 0){
			fMsr = 0;
			iRows = *(gBiclusters + uBicluster * colsmBiclusters + 0);
			iCols = *(gBiclusters + uBicluster * colsmBiclusters + 1);
		}
		__syncthreads();
		double subMsr = 0;
		for(ulong contLoop=0; contLoop < loopByThread && (uThread*loopByThread)+contLoop < iRows; contLoop++){
			ulong iRow = *(gBiclusters + uBicluster * colsmBiclusters + (2+(uThread*loopByThread)+contLoop));
			for(ulong contCols=0; contCols < iCols; contCols++){
				ulong iCol = *(gBiclusters + uBicluster * colsmBiclusters + (2+iRows+contCols));
				double element_aij = *(mGeneExpression + (iRow-1) * colsmExpression + (iCol-1)); // Element aij
				subMsr += (element_aij - *(gRowsMedia + uBicluster * rowsgRowsMedia + ((uThread*loopByThread)+contLoop)) - *(gColsMedia + uBicluster * colsgColsMedia + contCols) + *(gMediaBicluster + uBicluster)) * (element_aij - *(gRowsMedia + uBicluster * rowsgRowsMedia + ((uThread*loopByThread)+contLoop)) - *(gColsMedia + uBicluster * colsgColsMedia + contCols) + *(gMediaBicluster + uBicluster));
			}
		}
		atomicAdd(&fMsr, subMsr);
		__syncthreads();
		if(uThread == 0){
			fMsr /= (iRows*iCols);
			*(gResultMsr + uBicluster) = fMsr;
		}
	}
}

void introduceParameters(char **argv) {

	// PARAMETER 1: Biclusters file
	biclustersFile = argv[1];

	// PARAMETER 2: Matrix file
	matrixFile = "/home/principalpc/Tests/gMSR/Matrix/GDS4794/GDS4794.matrix";
	matrixFile = argv[2];

	//PARAMETER 3: Delta biclusters
	delta = atoi(argv[3]);

	// PARAMETER 4: Output maximum biclusters
	biclustersOutput = atoi(argv[4]);

	//PARAMETER 5: GPus number
	deviceCount = atoi(argv[5]);

	//PARAMETER 6: Output file
	outputFile = argv[6];

}

void fillBiclusters(ulong min, ulong biclustersPerChunkFill) {

	mBiclusters = (ulong *) malloc(biclustersPerChunkFill * cColsmBiclusters * sizeof(ulong));

	// 3) Fill the matrix
	for(ulong lBic = 0; lBic < biclustersPerChunkFill; lBic++){
		*(mBiclusters + lBic * cColsmBiclusters + 0) = aRows[lBic+min].size();
		*(mBiclusters + lBic * cColsmBiclusters + 1) = aCols[lBic+min].size();
		for(ulong lRows = 0; lRows < aRows[lBic+min].size() ; lRows++){
			*(mBiclusters + lBic * cColsmBiclusters + (2+lRows)) = (aRows[lBic+min])[lRows];

		}
		for(ulong lCols = 0; lCols < aCols[lBic+min].size() ; lCols++){
			*(mBiclusters + lBic * cColsmBiclusters + (2+aRows[lBic+min].size()+lCols)) = (aCols[lBic+min])[lCols];
		}
		for(ulong lRest = 2+aRows[lBic+min].size()+aCols[lBic+min].size(); lRest < cColsmBiclusters ; lRest++){
			*(mBiclusters + lBic * cColsmBiclusters + lRest) = 0;
		}
	}
}

void biclustersReader() {
	string line;
	ifstream myfile(biclustersFile.c_str());
	// 1) Read file
	if (myfile.is_open()) {
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
		cColsmBiclusters = 2+maxRows+maxCols;
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
	listMsr.sort();
	ofstream myfile(outputFile);
	if (myfile.is_open()) {
		list<Msr>::iterator it = listMsr.begin();
		myfile << "BICLUSTER_ID,MSR\n";
		for(int i = 0; i < listMsr.size() && i < biclustersOutput; i++){
			myfile << ((Msr)*it).uBicluster << "," << ((Msr)*it).dMsr << "\n";
			advance(it, 1);
		}
		myfile.close();
	} else cout << "Unable to save file";
}

tuple<ulong *, ulong, ulong> splitBiclusterMatrix(ulong biclustersPerChunkGpu, ulong min){
	ulong maxRowsSplit = 0, maxColsSplit = 0;
	ulong *cBiclusters = (ulong *) malloc(biclustersPerChunkGpu * cColsmBiclusters * sizeof(ulong));
	for(int r = 0; r < biclustersPerChunkGpu; r++){
		ulong iRow = *(mBiclusters + (r+min) * cColsmBiclusters + 0);
		ulong iCol = *(mBiclusters + (r+min) * cColsmBiclusters + 1);
		if(iRow > maxRowsSplit){
			maxRowsSplit = iRow;
		}
		if(iCol > maxColsSplit){
			maxColsSplit = iCol;
		}
		for(int c = 0; c < cColsmBiclusters; c++){
			*(cBiclusters + r * cColsmBiclusters + c) = *(mBiclusters + (r+min) * cColsmBiclusters + c);
		}
	}
	return make_tuple(cBiclusters,maxRowsSplit,maxColsSplit);
}

void prepareGpu1D(ulong lBlocks, ulong lThreads){
	int device;
	cudaGetDevice(&device);
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	lastBlocksGrid = lBlocks; // Case: 0 < lBlocks <= prop.maxGridSize[1]
	maxThreadsPerBlock = lThreads; // Case: 0 < lThreads <= prop.maxThreadsPerBlock
	maxIteratorGPU = 0;
	loopsByThread = 0;

	if(lastBlocksGrid > prop.maxGridSize[1]){ // Blocks Control: Not supported GPU with a for --> Split patterns in multiple for
		maxIteratorGPU = lastBlocksGrid / prop.maxGridSize[1];
		maxBlocksPerGrid = prop.maxGridSize[1];
		lastBlocksGrid = lastBlocksGrid - (maxIteratorGPU * prop.maxGridSize[1]);
	}

	if(maxThreadsPerBlock > prop.maxThreadsPerBlock){ // Threads Control: lThreads > prop.maxThreadsPerBlock && Supported GPU in a for
		loopsByThread = maxThreadsPerBlock / prop.maxThreadsPerBlock;
		if(maxThreadsPerBlock % prop.maxThreadsPerBlock != 0){
			loopsByThread++;
		}
		maxThreadsPerBlock = prop.maxThreadsPerBlock;
	}

	if(loopsByThread == 0){ // Debe realizar al menos un bucle en los kernels
		loopsByThread++;
	}
}

void threadsPerDevice(int idGpu, cudaStream_t s, ulong *chunks, ulong *biclustersPerChunkGpu, ulong *biclustersLastChunkGpu, double *mGeneExpression, ulong totalBiclustersPerGpu, mutex *m) {
	cudaSetDevice(idGpu);

	// 1) The minimum and maximum index of each GPU is calculated taking into account the chunks.
	ulong minIndexBicByGpu = 0;
	ulong maxIndexBicByGpu = (chunks[0]*biclustersPerChunkGpu[0])+biclustersLastChunkGpu[0];
	for(int i=1; i<=idGpu; i++){ // Multi-GPU
		minIndexBicByGpu += (chunks[i-1]*biclustersPerChunkGpu[i-1])+biclustersLastChunkGpu[i-1];
		maxIndexBicByGpu += (chunks[i]*biclustersPerChunkGpu[i])+biclustersLastChunkGpu[i];
	}
	maxIndexBicByGpu -= 1;

	// 2) Execute Chunks
	for(ulong largeScale = 0; largeScale < chunks[idGpu] ; largeScale++){
		ulong min = minIndexBicByGpu+(largeScale*biclustersPerChunkGpu[idGpu]);

		// 2.1) Result Array (MSR) to GPU global memory & CPU
		double *gResultMsr, *cResultMsr;
		cResultMsr = (double *) malloc(biclustersPerChunkGpu[idGpu] * sizeof(double));
		cudaMalloc((void **)&gResultMsr, biclustersPerChunkGpu[idGpu] * sizeof(double));
		cudaMemset(gResultMsr, 0, biclustersPerChunkGpu[idGpu] * sizeof(double));

		// 2.2) Dynamic split of the Bicluster Matrix file to GPU devices.
		ulong *gBiclusters, *cBiclusters, maxRows, maxCols;
		tie(cBiclusters,maxRows,maxCols) = splitBiclusterMatrix(biclustersPerChunkGpu[idGpu], min);
		cudaMalloc((void**) &gBiclusters, biclustersPerChunkGpu[idGpu] * cColsmBiclusters * sizeof(ulong));
		cudaMemcpy(gBiclusters, cBiclusters, biclustersPerChunkGpu[idGpu] * cColsmBiclusters * sizeof(ulong), cudaMemcpyHostToDevice);

		// 2.3) Calculate rows media and total bicluster
		double *gMediaBicluster, *gRowsMedia, *gColsMedia;
		cudaMalloc((void **)&gMediaBicluster, biclustersPerChunkGpu[idGpu] * sizeof(double));
		cudaMemset(gMediaBicluster, 0, biclustersPerChunkGpu[idGpu] * sizeof(double));
		cudaMalloc((void **)&gRowsMedia, biclustersPerChunkGpu[idGpu] * maxRows * sizeof(double));
		cudaMemset(gRowsMedia, 0, biclustersPerChunkGpu[idGpu] * maxRows * sizeof(double));
		cudaMalloc((void **)&gColsMedia, biclustersPerChunkGpu[idGpu] * maxCols * sizeof(double));
		cudaMemset(gColsMedia, 0, biclustersPerChunkGpu[idGpu] * maxCols * sizeof(double));

		prepareGpu1D(biclustersPerChunkGpu[idGpu],maxRows);
		for(int i=1; i <= maxIteratorGPU; i++){
			calculateRowsTotalMedia<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, maxRows, biclustersPerChunkGpu[idGpu], i, maxBlocksPerGrid);
		}
		calculateRowsTotalMedia<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, maxRows, biclustersPerChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

		prepareGpu1D(biclustersPerChunkGpu[idGpu],maxCols);
		for(int i=1; i <= maxIteratorGPU; i++){
			calculateColsMedia<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, mGeneExpression, gBiclusters, gColsMedia, maxCols, biclustersPerChunkGpu[idGpu], i, maxBlocksPerGrid);
		}
		calculateColsMedia<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, mGeneExpression, gBiclusters, gColsMedia, maxCols, biclustersPerChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

		prepareGpu1D(biclustersPerChunkGpu[idGpu],maxRows);
		for(int i=1; i <= maxIteratorGPU; i++){
			calculateMsr<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, gResultMsr, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, gColsMedia, maxRows, maxCols, biclustersPerChunkGpu[idGpu], i, maxBlocksPerGrid);
		}
		calculateMsr<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, gResultMsr, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, gColsMedia, maxRows, maxCols, biclustersPerChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

		cudaMemcpy(cResultMsr, gResultMsr, biclustersPerChunkGpu[idGpu] * sizeof(double), cudaMemcpyDeviceToHost);

		for(ulong r=0; r < biclustersPerChunkGpu[idGpu]; r++){
			double dMsr = *(cResultMsr + r);
			if(dMsr >= 0 && dMsr <= delta){
				m->lock();
				listMsr.push_back(Msr(min+r+1,dMsr));
				m->unlock();
			}
		}

		cudaFree(gResultMsr);
		cudaFree(gBiclusters);
		cudaFree(gMediaBicluster);
		cudaFree(gRowsMedia);
		cudaFree(gColsMedia);
		free(cResultMsr);
		free(cBiclusters);
	}

	// 3) Execute last chunk (last biclusters)
	ulong min = minIndexBicByGpu+(chunks[idGpu]*biclustersPerChunkGpu[idGpu]);

	// 3.1) Result Array (MSR) to GPU global memory & CPU
	double *gResultMsr, *cResultMsr;
	cResultMsr = (double *) malloc(biclustersLastChunkGpu[idGpu] * sizeof(double));
	cudaMalloc((void **)&gResultMsr, biclustersLastChunkGpu[idGpu] * sizeof(double));
	cudaMemset(gResultMsr, 0, biclustersLastChunkGpu[idGpu] * sizeof(double));

	// 3.2) Dynamic split of the Bicluster Matrix file to GPU devices.
	ulong *gBiclusters, *cBiclusters, maxRows, maxCols;
	tie(cBiclusters,maxRows,maxCols) = splitBiclusterMatrix(biclustersLastChunkGpu[idGpu], min);
	cudaMalloc((void**) &gBiclusters, biclustersLastChunkGpu[idGpu] * cColsmBiclusters * sizeof(ulong));
	cudaMemcpy(gBiclusters, cBiclusters, biclustersLastChunkGpu[idGpu] * cColsmBiclusters * sizeof(ulong), cudaMemcpyHostToDevice);

	// 3.3) Calculate rows media and total bicluster
	double *gMediaBicluster, *gRowsMedia, *gColsMedia;
	cudaMalloc((void **)&gMediaBicluster, biclustersLastChunkGpu[idGpu] * sizeof(double));
	cudaMemset(gMediaBicluster, 0, biclustersLastChunkGpu[idGpu] * sizeof(double));
	cudaMalloc((void **)&gRowsMedia, biclustersLastChunkGpu[idGpu] * maxRows * sizeof(double));
	cudaMemset(gRowsMedia, 0, biclustersLastChunkGpu[idGpu] * maxRows * sizeof(double));
	cudaMalloc((void **)&gColsMedia, biclustersLastChunkGpu[idGpu] * maxCols * sizeof(double));
	cudaMemset(gColsMedia, 0, biclustersLastChunkGpu[idGpu] * maxCols * sizeof(double));

	prepareGpu1D(biclustersLastChunkGpu[idGpu],maxRows);
	for(int i=1; i <= maxIteratorGPU; i++){
		calculateRowsTotalMedia<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, maxRows, biclustersLastChunkGpu[idGpu], i, maxBlocksPerGrid);
	}
	calculateRowsTotalMedia<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, maxRows, biclustersLastChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

	prepareGpu1D(biclustersLastChunkGpu[idGpu],maxCols);
	for(int i=1; i <= maxIteratorGPU; i++){
		calculateColsMedia<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, mGeneExpression, gBiclusters, gColsMedia, maxCols, biclustersLastChunkGpu[idGpu], i, maxBlocksPerGrid);
	}
	calculateColsMedia<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, mGeneExpression, gBiclusters, gColsMedia, maxCols, biclustersLastChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

	prepareGpu1D(biclustersLastChunkGpu[idGpu],maxRows);
	for(int i=1; i <= maxIteratorGPU; i++){
		calculateMsr<<<maxBlocksPerGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, gResultMsr, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, gColsMedia, maxRows, maxCols, biclustersLastChunkGpu[idGpu], i, maxBlocksPerGrid);
	}
	calculateMsr<<<lastBlocksGrid, maxThreadsPerBlock,0,s>>>(loopsByThread, idGpu, min, gResultMsr, mGeneExpression, gBiclusters, gMediaBicluster, gRowsMedia, gColsMedia, maxRows, maxCols, biclustersLastChunkGpu[idGpu], maxIteratorGPU+1, maxBlocksPerGrid);

	cudaMemcpy(cResultMsr, gResultMsr, biclustersLastChunkGpu[idGpu] * sizeof(double), cudaMemcpyDeviceToHost);

	for(ulong r=0; r < biclustersLastChunkGpu[idGpu]; r++){
		double dMsr = *(cResultMsr + r);
		if(dMsr >= 0 && dMsr <= delta){
			m->lock();
			listMsr.push_back(Msr(min+r+1,dMsr));
			m->unlock();
		}
	}

	cudaFree(gResultMsr);
	cudaFree(gBiclusters);
	cudaFree(gMediaBicluster);
	cudaFree(gRowsMedia);
	cudaFree(gColsMedia);
	free(cResultMsr);
	free(cBiclusters);

	cudaFree(mExpression);
}

void runAlgorithm() {

	struct sysinfo myinfo;
	sysinfo(&myinfo);
	size_t freeRam = (3*(myinfo.freeram+myinfo.freeswap))/4;
	size_t sizeBiclusters = cRowsmBiclusters * cColsmBiclusters * sizeof(ulong);
	ulong chunksRam = sizeBiclusters/freeRam;
	long biclustersPerChunkRam = 0, lastbiclustersPerChunkRam = 0;
	if(chunksRam == 0){
		lastbiclustersPerChunkRam = cRowsmBiclusters;
	} else {
		biclustersPerChunkRam = cRowsmBiclusters/(chunksRam+1);
		lastbiclustersPerChunkRam = cRowsmBiclusters - (chunksRam*biclustersPerChunkRam);
	}

	// Limit RAM blocks
	for(ulong iChunk = 0; iChunk < chunksRam ; iChunk++){
		ulong min = iChunk*biclustersPerChunkRam;
		fillBiclusters(min,biclustersPerChunkRam);

		// Prepare GPU large-scale data
		// Chunks: Number of a GPU runs due to the input arrays are larger than the available memory of the GPU.
		// biclustersPerChunkGpu: How many biclusters per chunk can process a GPU device.
		cudaStream_t s[deviceCount];
		thread threads[deviceCount];
		ulong *chunks = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong *biclustersPerChunkGpu = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong *biclustersLastChunkGpu = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong totalBiclustersPerGpu = biclustersPerChunkRam/deviceCount;
		ulong restBiclustersLastGpu = biclustersPerChunkRam%deviceCount;

		for(int i=0; i<deviceCount; i++){
			cudaSetDevice(i);
			cudaStreamCreate(&s[i]);
			struct cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			double availableMemory = ((3*prop.totalGlobalMem)/4 - (cRowsmExpression * cColsmExpression * sizeof(double))); // Gene expression matrix (mExpression)
			double sizeResult = (totalBiclustersPerGpu * sizeof(double)); //Result Matrix (MSR)
			double sizeBiclusters = (totalBiclustersPerGpu * cColsmBiclusters * sizeof(ulong)); // Bicluster Matrix File
			double sizeMediaBicluster = (totalBiclustersPerGpu * sizeof(double)); // gMediaBicluster
			double sizeMediaRows = (totalBiclustersPerGpu * cRowsmBiclusters * sizeof(double)); // gRowsMedia
			double sizeMediaCols = (totalBiclustersPerGpu * cColsmBiclusters * sizeof(double)); // gColsMedia
			double chunkValue = ((sizeResult+sizeBiclusters+sizeMediaBicluster+sizeMediaRows+sizeMediaCols)/availableMemory)+1;

			chunks[i] = chunkValue;
			biclustersPerChunkGpu[i] = totalBiclustersPerGpu / chunkValue;
			biclustersLastChunkGpu[i] = totalBiclustersPerGpu - (biclustersPerChunkGpu[i] * chunks[i]);
			if(deviceCount > 1 && restBiclustersLastGpu!=0 && i==deviceCount-1){ // multi-GPU support: If it is the last GPU and there are still more biclusters to execute, the last GPU execute the biclusters exceded.
				biclustersLastChunkGpu[i] += restBiclustersLastGpu;
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
			cudaMalloc((void**) &mGeneExpression, cRowsmExpression * cColsmExpression * sizeof(double));
			cudaMemcpy(mGeneExpression, mExpression, cRowsmExpression * cColsmExpression * sizeof(double), cudaMemcpyHostToDevice);
			threads[i] = thread(threadsPerDevice, i, s[i], chunks, biclustersPerChunkGpu, biclustersLastChunkGpu, mGeneExpression, totalBiclustersPerGpu, &m);
		}

		for(auto& th: threads){
			th.join();
		}

		free(mBiclusters);
	}

	//Last RAM block
	if(lastbiclustersPerChunkRam>0){
		ulong min = chunksRam*biclustersPerChunkRam;
		fillBiclusters(min,lastbiclustersPerChunkRam);

		// Prepare GPU large-scale data
		// Chunks: Number of a GPU runs due to the input arrays are larger than the available memory of the GPU.
		// biclustersPerChunkGpu: How many biclusters per chunk can process a GPU device.

		cudaStream_t s[deviceCount];
		thread threads[deviceCount];
		ulong *chunks = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong *biclustersPerChunkGpu = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong *biclustersLastChunkGpu = (ulong *) malloc(deviceCount * sizeof(ulong));
		ulong totalBiclustersPerGpu = lastbiclustersPerChunkRam/deviceCount;
		ulong restBiclustersLastGpu = lastbiclustersPerChunkRam%deviceCount;

		for(int i=0; i<deviceCount; i++){
			cudaSetDevice(i);
			cudaStreamCreate(&s[i]);
			struct cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			double availableMemory = ((3*prop.totalGlobalMem)/4 - (cRowsmExpression * cColsmExpression * sizeof(double))); // Gene expression matrix (mExpression)
			double sizeResult = (totalBiclustersPerGpu * sizeof(double)); //Result Matrix (MSR)
			double sizeBiclusters = (totalBiclustersPerGpu * cColsmBiclusters * sizeof(ulong)); // Bicluster Matrix File
			double sizeMediaBicluster = (totalBiclustersPerGpu * sizeof(double)); // gMediaBicluster
			double sizeMediaRows = (totalBiclustersPerGpu * cRowsmBiclusters * sizeof(double)); // gRowsMedia
			double sizeMediaCols = (totalBiclustersPerGpu * cColsmBiclusters * sizeof(double)); // gColsMedia
			double chunkValue = ((sizeResult+sizeBiclusters+sizeMediaBicluster+sizeMediaRows+sizeMediaCols)/availableMemory)+1;

			chunks[i] = chunkValue;
			biclustersPerChunkGpu[i] = totalBiclustersPerGpu / chunkValue;
			biclustersLastChunkGpu[i] = totalBiclustersPerGpu - (biclustersPerChunkGpu[i] * chunks[i]);
			if(deviceCount > 1 && restBiclustersLastGpu!=0 && i==deviceCount-1){ // multi-GPU support: If it is the last GPU and there are still more biclusters to execute, the last GPU execute the biclusters exceded.
				biclustersLastChunkGpu[i] += restBiclustersLastGpu;
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
			cudaMalloc((void**) &mGeneExpression, cRowsmExpression * cColsmExpression * sizeof(double));
			cudaMemcpy(mGeneExpression, mExpression, cRowsmExpression * cColsmExpression * sizeof(double), cudaMemcpyHostToDevice);
			threads[i] = thread(threadsPerDevice, i, s[i], chunks, biclustersPerChunkGpu, biclustersLastChunkGpu, mGeneExpression, totalBiclustersPerGpu, &m);
		}

		for(auto& th: threads){
			th.join();
		}

		free(mBiclusters);
	}
}

int main(int argc, char** argv) {

	introduceParameters(argv);
	readerMatrix();
	biclustersReader();

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	runAlgorithm();

	// Print DELTA BICLUSTERS
	cout << "##################" << endl;
	cout << "GMSR (v1_2) INFO:" << endl;
	cout << "##################" << endl;
	cout << "Delta filter: " << delta << endl;
	cout << "Results save in: " << outputFile << endl;
	cout << endl << endl;

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	saveFile();

	return 0;
}


