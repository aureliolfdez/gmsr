//============================================================================
// Name        : seqMsr.cpp
// Author      : Aurelio Lopez-Fernandez
// Version     :
// Copyright   : Your copyright notice
// Description : Esta version calcula primeramente las medias y las almacena en vectores. Despues simplemente calcula el MSR en funcion de las medias almacenadas.
//============================================================================

#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>
#include <sstream>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <math.h>
using namespace std;

//introduceParameters
string biclustersFile, matrixFile, outputFile;
double delta;

//Matrix biclusters
ulong rowsmBiclusters, colsmBiclusters, *mBiclusters;

//Matrix expression gene
ulong rowsmExpression, colsmExpression;
double *mExpression;

// Delta biclusters
struct compMsr {
    bool operator() (const pair<ulong, double>& elem1, const pair<ulong, double>& elem2) const {
    	return elem1.second < elem2.second;
    }
};
set<pair<ulong, double>, compMsr> setMsr;

void introduceParameters() {

	// PARAMETER 1: Biclusters file
	biclustersFile = "/home/principalpc/G-MSR/Results/prueba.csv";

	// PARAMETER 2: Matrix file
	matrixFile = "/home/principalpc/G-MSR/Matrix/yeast.matrix";

	//PARAMETER 3: OUTPUT
	delta = 1000;

	//PARAMETER 4: Output file
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
			rowsmBiclusters++;
		}
		myfile.close();

		// 2) Create matrix
		colsmBiclusters = 2+maxRows+maxCols;
		mBiclusters = (ulong *) malloc(rowsmBiclusters * colsmBiclusters * sizeof(ulong));

		// 3) Fill the matrix
		for(ulong lBic = 0; lBic < rowsmBiclusters; lBic++){
			*(mBiclusters + lBic * colsmBiclusters + 0) = aRows[lBic].size();
			*(mBiclusters + lBic * colsmBiclusters + 1) = aCols[lBic].size();
			for(ulong lRows = 0; lRows < aRows[lBic].size() ; lRows++){
				*(mBiclusters + lBic * colsmBiclusters + (2+lRows)) = (aRows[lBic])[lRows];
			}
			for(ulong lCols = 0; lCols < aCols[lBic].size() ; lCols++){
				*(mBiclusters + lBic * colsmBiclusters + (2+aRows[lBic].size()+lCols)) = (aCols[lBic])[lCols];
			}
			for(ulong lRest = 2+aRows[lBic].size()+aCols[lBic].size(); lRest < colsmBiclusters ; lRest++){
				*(mBiclusters + lBic * colsmBiclusters + lRest) = 0;
			}
		}
	} else {
		cout << "Unable to open file " << endl;
	}
}

void readerMatrix() {

	// 1) Prepare Matrix from file
	rowsmExpression = 0;
	colsmExpression = 0;
	vector<string> rowsArray_Aux;
	string line;
	ifstream myfile(matrixFile.c_str());
	if (myfile.is_open()) {
		// 2.1) Get number of rows
		while (getline(myfile, line)) {
			rowsArray_Aux.push_back(line);
			// 2.2) Get number of columns
			if (rowsmExpression == 0) {
				for (int k = line.size() - 1; k >= 0; k--) {
					if (line[k] == ',') {
						colsmExpression++;
					}
				}
				colsmExpression++;
			}
			rowsmExpression++;
		}
		myfile.close();

		// 2) Create matrix
		mExpression = (double *) malloc(rowsmExpression * colsmExpression * sizeof(double));

		// 3) Fill matrix
		for (ulong j = 0; j < rowsmExpression; j++) {
			string row = rowsArray_Aux[j];
			stringstream ss(row);
			double i;
			int contCols = 0;
			while (ss >> i) {
				*(mExpression + j * colsmExpression + contCols) = i;
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

double calculateMsr(ulong iBicluster){
	ulong iRows = *(mBiclusters + iBicluster * colsmBiclusters + 0);
	ulong iCols = *(mBiclusters + iBicluster * colsmBiclusters + 1);

	// 1) For each bicluster - Calculate rows media and total bicluster
	double mediaBicluster = 0;
	double *mediaRows = (double *) malloc(iRows * sizeof(double));
	for(ulong contRow=0; contRow < iRows ; contRow++){
		ulong iRow = *(mBiclusters + iBicluster * colsmBiclusters + (2+contRow));
		double media = 0;
		for(ulong contMedia=0; contMedia < iCols; contMedia++){
			ulong iCol = *(mBiclusters + iBicluster * colsmBiclusters + (2+iRows+contMedia));
			media += *(mExpression + iRow * colsmExpression + iCol);
			mediaBicluster += *(mExpression + iRow * colsmExpression + iCol);
		}
		media /= iCols;
		*(mediaRows + contRow) = media;
	}
	mediaBicluster /= (iRows*iCols);

	// 2) For each bicluster - Calculate cols media
	double *mediaCols = (double *) malloc(iCols * sizeof(double));
	for(ulong contCol=0; contCol < iCols ; contCol++){
		ulong iCol = *(mBiclusters + iBicluster * colsmBiclusters + (2+iRows+contCol));
		double media = 0;
		for(ulong contRow=0; contRow < iRows; contRow++){
			ulong iRow = *(mBiclusters + iBicluster * colsmBiclusters + (2+contRow));
			media += *(mExpression + iRow * colsmExpression + iCol);
		}
		media /= iRows;
		*(mediaCols + contCol) = media;
	}

	// 3) Calculate MSR by element
	double fMsr = 0;
	for(ulong contRow=0; contRow < iRows ; contRow++){
		ulong iRow = *(mBiclusters + iBicluster * colsmBiclusters + (2+contRow));
		for(ulong contCols=0; contCols < iCols; contCols++){
			ulong iCol = *(mBiclusters + iBicluster * colsmBiclusters + (2+iRows+contCols));
			double element_aij = *(mExpression + iRow * colsmExpression + iCol); // Element aij
			fMsr += pow(element_aij - *(mediaRows + contRow) - *(mediaCols + contCols) + mediaBicluster,2);
		}
	}
	fMsr /= (iRows*iCols);
	return fMsr;
}

void runAlgorithm() {

	// Ordered and filtered delta biclusters
	for(ulong r=0; r < rowsmBiclusters; r++){
		double fMsr = calculateMsr(r);
		if(fMsr >= 0 && fMsr <= delta){
			pair<ulong, double> x = make_pair(r+1,fMsr);
			setMsr.insert(x);
		}
	}
}

int main() {
	introduceParameters();
	readerMatrix();
	biclustersReader();
	runAlgorithm();

	// Print DELTA BICLUSTERS
	cout << "##################" << endl;
	cout << "SEQ_MSR (v1) INFO:" << endl;
	cout << "##################" << endl;
	cout << "Delta filter: " << delta << endl;
	cout << "Results save in: " << outputFile << endl;
	cout << endl << endl;
	saveFile();

	return 0;
}


