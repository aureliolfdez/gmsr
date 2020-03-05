# gMSR: A multi-GPU algorithm to accelerate the biclusters validation

## Introduction to gMSR
Working...

## Compilation
Go to src/CUDA folder and executes the following commands to compile:
```
nvcc -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "src/CUDA/gMsr.d" "src/CUDA/gMsr.cu"
nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "src/CUDA/gMsr.o" "src/gMSR/gMsr.cu"
```


## Execution
### 1. Input parameters
1. biclustersFile (Character string): Absolute path of the input biclusters dataset file.
2. matrixFile (Character string): Absolute path of the input gene-expression matrix dataset file.
3. delta (integer number): Maximum MSR value allowed.
4. biclustersOutput (integer number): Maximum number of biclusters generated.
5. deviceCount (integer number): Number of GPU devices you want to use.
2. outputFile (Character string): Absolute path of the output file.

### 2. Execute
_./gMsr [biclustersFile] [matrixFile] [delta] [biclustersOutput] [deviceCount]_[outputFile]

```
./gMsr /home/MyUser/Tests/bicDataset.csv /home/MyUser/Tests/geneMatrix.matrix 2000 100 2 /home/MyUser/Tests/output.csv
```

## Authors
* [Aurelio Lopez-Fernandez](mailto:alopfer1@upo.es) - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Domingo Rodriguez-Baena - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)

## Contact
If you have comments or questions, or if you would like to contribute to the further development of gBiBit, please send us an email at alopfer1@upo.es

## License
This projected is licensed under the terms of the GNU General Public License v3.0.
