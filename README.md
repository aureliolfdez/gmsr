# gMSR: A multi-GPU algorithm to accelerate a massive validation of biclusters

Biclustering is nowadays one of the most widely used Machine Learning techniques to discover local patterns in datasets from different areas such as Energy Consumption, Marketing, Social Networks or Bioinformatics, among them. Particularly in Bioinformatics, Biclustering techniques have become extremely time-consuming, also being huge the number of results generated, due to the continuous increase in the size of the databases over the last few years. For this reason, validation techniques must be adapted to this new environment in order to help researchers focus their efforts on a specific subset of results in an efficient, fast and reliable way. The aforementioned situation may well be considered as Big Data context. In this sense, multiple Machine Learning techniques have been implemented by the application of GPU technology and CUDA architecture to accelerate the processing of large databases. However, as far as we know, this technology has not yet been applied to any bicluster validation technique. In this work, a multi-GPU version of one of the most used bicluster validation measure, MSR, is presented. It takes advantage of all the hardware and memory resources offered by GPU devices. Due to this, gMSR is able to validate a massive number of biclusters in any Biclustering-based study within a Big Data context.

## Requirements
It is recommended to have installed at least:
1. CUDA 11.0 Toolkit (11.0.171).
2. NVIDIA Driver version: 450.36.06
3. Host compile executable: gcc and g++ on Linux, clang and clang++ on Mac OS X or cl.exe on Windows.

## Compilation
1. Firstly, get the version of the compute capability (CC) of your NVIDIA graphics card: https://developer.nvidia.com/cuda-gpus
2. Go to src/CUDA folder.
3. Replace compute_61 and sm_61 to your compute capability version and the host compiler according to your operating system in the following commands.

Example of a compilation on GNU/Linux with a CC version of 6.1:
```
nvcc -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "src/CUDA/gMsr.d" "src/CUDA/gMsr.cu"
nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "src/CUDA/gMsr.o" "src/CUDA/gMsr.cu"
nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -link -o "gMsr" ./src/CUDA/gMsr.o
```
Running the above commands will generate an executable called gMSR.

## Execution
### 1. Input parameters
1. biclustersFile (Character string): Absolute path of the input biclusters dataset file.
2. matrixFile (Character string): Absolute path of the input gene-expression matrix dataset file.
3. delta (integer number): Maximum MSR value allowed.
4. biclustersOutput (integer number): Number of biclusters to be included in the ordered list returned as result.
5. deviceCount (integer number): Number of GPU devices you want to use.
2. outputFile (Character string): Absolute path of the output file.

### 2. Execute
_./gMsr [biclustersFile] [matrixFile] [delta] [biclustersOutput] [deviceCount]_[outputFile]

```
./gMsr /home/MyUser/Tests/bicDataset.csv /home/MyUser/Tests/geneMatrix.matrix 2000 100 2 /home/MyUser/Tests/output.csv
```

## Authors
* [Aurelio Lopez-Fernandez](mailto:alopfer1@upo.es) - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Domingo S. Rodriguez-Baena - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)
* Francisco Gómez-Vela - [DATAi Research Group (Pablo de Olavide University)](http://www.upo.es/investigacion/datai)

## Contact
If you have comments or questions, or if you would like to contribute to the further development of gBiBit, please send us an email at alopfer1@upo.es

## License
This projected is licensed under the terms of the GNU General Public License v3.0.
