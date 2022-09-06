# ParallelNeighborSearch

This work allows for furthest or nearest parallel neighbor searches in a HPC cluster with high efficiency. It is based in the library [MLPack](https://www.mlpack.org/), from authors R.R. Curtin, M. Edel, M. Lozhnikov, Y. Mentekidis, S. Ghaisas and S. Zhang.

## Requirements
To compile and execute this library, the following packages are required:
* GNU Make(>=v3.82)
* GCC (>=v.8.3.0)
* OpenMPI (>=3.1.4)
* Armadillo (>=10.8.2)
* MLPack (>=3.4.2)

It is possible that older versions of these packages may work, but they may not be able to provide correct results. It's also possible that different implementations of MPI may work but the one used to test the project was OpenMPI and therefore it's the recommended one.

## Compilation
Download the code using Git or directly from GitHub and use make to automatically compile the code. In the 'target' folder a new file called parallelSearch will be created that can be used to execute the searches.

## Execution
To execute the algorithm, use the following command : 
```
foo@bar:target$ mpirun -np <processes> parallelSearch <options>
```
where \<processes\> is the number of processes that will exist in the MPI group.

The possible options are :

* **-q** \<inputFile\> : Path of the dataset that contains the queries whose neighbors are to be obtained. The format must be CSV. 
* **-r** \<inputFile\> : Path of the dataset that contains the reference set from which the neighbors are  obtained. The format must be CSV. 
* **-n** \<neighbors\> : Number of neighbors to be obtained for each query.
* **-d** \<decompositionType\> : Type of decomposition that is to be used. The  possibilities are 'query' or 'reference'.
* **-m** \<method\> : Method to be used to find the neighbors. The possibilities are 'fns' or 'nns'.
* **-o** \<finalFile\> (optional) : File in which the results are to be stored. If  this argument is not passed, the results will be stored in a file called "results.txt".

## License

ParallelNeighborSearch is free software and as such it is distributed under the [MIT License](licenses/MIT.txt).