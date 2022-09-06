#include "ParallelNeighborSearch.hpp"

const int PROCESS_PER_NODE = 16;
const bool NEAREST_NEIGHBOR = true;
const bool FURTHEST_NEIGHBOR = false;

ParallelNeighborSearch::ParallelNeighborSearch(std::string querySetFilename, std::string referenceSetFilename, DecompositionType initialDtype){
    querySet = readCSV(querySetFilename).t();
    referenceSet = readCSV(referenceSetFilename).t();
    dType = initialDtype;
}

arma::mat ParallelNeighborSearch::readCSV(const std::string &filename, const std::string &delimeter)
{
    std::ifstream csv(filename);
    std::vector<std::vector<double>> datas;
    double a;

    for(std::string line; std::getline(csv, line); ) {

        std::vector<double> data;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);
        
        while (end != std::string::npos) {
            try {
              a = std::stod(line.substr(start, end - start));
              data.push_back(a);
            }
            catch (const std::invalid_argument& e )
            {
              data.push_back(0);
            }
            
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        try {
          a = std::stod(line.substr(start, end));
          data.push_back(a);
        }
        catch (const std::invalid_argument& e )
        {
          data.push_back(0);
        }
        datas.push_back(data);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(datas.size(), datas[0].size());

    for (int i=0; i<datas.size(); i++) {
        arma::mat r(datas[i]);
        data_mat.row(i) = r.t();
    }

    return data_mat;
}

void ParallelNeighborSearch::serializePartialResults ( arma::Mat<size_t> &neighbors,arma::mat &distances, int partialNeighbors[], double partialDistances[], int neigh, int rank, int size){
  int division  = ceil((querySet.n_cols+0.0)/size);
  for (int i = 0; i<querySet.n_cols; i++){
    for (int j=0; j<neigh;j++){
      partialNeighbors[i*neigh + j] = rank * division + neighbors.col(i)(j);
      partialDistances[i*neigh + j] = distances.col(i)(j);
    }
  } 
}

void ParallelNeighborSearch::getPartialResults(double partialDistances[], int partialNeighbors[], arma::mat &distances, arma::Mat<size_t> &neighbors, int neigh, MPI_Comm comm, bool nearest){
    int * totalNeighbors;
    double * totalDistances;
    int newRank, newSize;
    int ncols  = querySet.n_cols;
    MPI_Comm_size(comm, &newSize);
    MPI_Comm_rank(comm, &newRank);

    int totalSize = neigh * ncols * newSize;

    if (newRank == 0) {
        totalNeighbors = (int *) malloc(sizeof(int) *totalSize);
        if (totalNeighbors == nullptr) {
            cout << "Error de Memoria Vecinos  Totales\n";
            MPI::COMM_WORLD.Abort(-1);
        }
        totalDistances = (double *) malloc (sizeof(double) * totalSize);
        if (totalDistances == nullptr) {
            cout << "Error de Memoria Distancia Total\n";
            MPI::COMM_WORLD.Abort(-1);
        }
    }

    MPI_Gather(partialNeighbors, neigh*ncols, MPI_INT, totalNeighbors, neigh*ncols, MPI_INT, 0, comm);
    MPI_Gather(partialDistances, neigh*ncols, MPI_DOUBLE, totalDistances, neigh*ncols, MPI_DOUBLE, 0, comm);

    if (newRank == 0) {
        priority_queue<pointDistance, vector<pointDistance>, pointDistanceNearestComparator> pqueueNNS;
        priority_queue<pointDistance, vector<pointDistance>, pointDistanceFurthestComparator> pqueueFNS;
        pointDistance pair;
        int start;
        bool first;
        if (nearest){
            for (int i=0; i<ncols;i++){
                start = i*neigh;
                first = true;
      
                while(start<totalSize){
                    for (int j = 0; j< neigh;j++){
                        pair = std::make_pair(totalNeighbors[start+j], totalDistances[start+j]);
                        pqueueNNS.push(pair);
                        if (!first) {
                            pqueueNNS.pop();
                        }
                    }
                    first = false;
                    start += ncols * neigh;
                }

                for (int j =neigh-1; j>=0;j--){
                    neighbors.col(i)(j) = pqueueNNS.top().first;
                    distances.col(i)(j) = pqueueNNS.top().second;
                    pqueueNNS.pop();
                }
            }

            delete[]totalNeighbors;
            delete[]totalDistances;
        }
    else {
        for (int i=0; i<ncols;i++){
            start = i*neigh;
            first = true;
                
            while(start<totalSize){
                for (int j = 0; j< neigh;j++){
                    pair = std::make_pair(totalNeighbors[start+j], totalDistances[start+j]);
                    pqueueFNS.push(pair);
                    if (!first) {
                       pqueueFNS.pop();
                    }
                }
                first = false;
                start += ncols * neigh;
            }

            for (int j =neigh-1; j>=0;j--){
                neighbors.col(i)(j) = pqueueFNS.top().first;
                distances.col(i)(j) = pqueueFNS.top().second;
                pqueueFNS.pop();
            }
        }

        delete[]totalNeighbors;
        delete[]totalDistances;
    }
    }
}

void ParallelNeighborSearch::setDecompositionType(DecompositionType newDType){
    dType = newDType;
}

void ParallelNeighborSearch::domainDecomposition(arma::mat &set, int rank, int size,  arma::mat &setProc){
    int fin, ncols, division;

    ncols = set.n_cols;
    division = ceil((ncols+0.0)/size);

    if(rank==size-1){
        fin = ncols - division * (size-1);
    }
    else{
        fin = division;
    }

    setProc = set.cols(rank*division, rank*division+fin-1);
}

void ParallelNeighborSearch::referenceSetPostProcessing(arma::Mat<size_t> &neighbors, arma::mat &distances, int rank, int size, bool nearest, int neigh){
    MPI_Comm c1, c2;
    int * partialNeighbors = (int *) malloc(sizeof(int) * neigh * querySet.n_cols);
    if (partialNeighbors == nullptr){
        cout << "Error de Memoria Vecino Parcial\n";
        MPI::COMM_WORLD.Abort(-1);
    }
    double * partialDistances = (double *) malloc(sizeof(double) * neigh * querySet.n_cols);
    if (partialDistances == nullptr){
        cout << "Error de Memoria Distancia Parcial\n";
        MPI::COMM_WORLD.Abort(-1);
    }

    serializePartialResults(neighbors, distances,partialNeighbors, partialDistances, neigh, rank, size);

    if (size <PROCESS_PER_NODE*2) {
        getPartialResults(partialDistances, partialNeighbors, distances, neighbors, neigh,MPI_COMM_WORLD, nearest);
        delete[] partialNeighbors;
        delete[] partialDistances;
        }
    else {
        MPI_Comm_split(MPI::COMM_WORLD,floor(rank/PROCESS_PER_NODE), rank, &c1);
        MPI_Comm_split(MPI::COMM_WORLD,rank%PROCESS_PER_NODE, rank, &c2);
        getPartialResults(partialDistances, partialNeighbors, distances, neighbors, neigh,c1, nearest);
        MPI::COMM_WORLD.Barrier();
        if(rank%PROCESS_PER_NODE == 0) {
            serializePartialResults(neighbors, distances,partialNeighbors, partialDistances, neigh, rank, size);
            getPartialResults(partialDistances, partialNeighbors, distances, neighbors, neigh, c2, nearest);
        }
        MPI::COMM_WORLD.Barrier();
        delete[] partialNeighbors;
        delete[] partialDistances;
    }

}

void ParallelNeighborSearch::querySetPostProcessing(arma::Mat<size_t> &neighbors, arma::mat &distances,arma::Mat<size_t> &localNeighbors, arma::mat &localDistances, int rank, int neigh, int procNCols){
    int division, end;
    
    int * partialNeighbors = (int *) malloc(sizeof(int) * neigh * procNCols);
    if (partialNeighbors == nullptr){
        cout << "Error de Memoria Vecino Parcial\n";
        MPI::COMM_WORLD.Abort(-1);
    }
    double * partialDistances = (double *) malloc(sizeof(double) * neigh * procNCols);
    if (partialDistances == nullptr){
        cout << "Error de Memoria Distancia Parcial\n";
        MPI::COMM_WORLD.Abort(-1);
    }

    for (int i = 0; i<procNCols; i++){
        for (int j=0; j<neigh;j++){
            partialNeighbors[i*neigh + j] = neighbors.col(i)(j);
            partialDistances[i*neigh + j] = distances.col(i)(j);
        }
    }

    int * totalNeighbors;
    double * totalDistances;

    if (rank == 0) {
        totalNeighbors = (int *) malloc(sizeof(int) * querySet.n_cols * neigh);
        if (totalNeighbors == nullptr) {
            cout << "Error de Memoria Vecinos  Totales\n";
            MPI::COMM_WORLD.Abort(-1);
        }
        totalDistances = (double *) malloc (sizeof(double) * querySet.n_cols * neigh);
        if (totalDistances == nullptr) {
            cout << "Error de Memoria Distancia Total\n";
            MPI::COMM_WORLD.Abort(-1);
        }
    }

    division = ceil((querySet.n_cols+0.0)/size);
    end = querySet.n_cols - division * (size-1);

    int receive_c[size];
    int receive_d[size];

    receive_c[0] = neigh *division;
    receive_d[0] = 0;

    for (int i=1;i<size;i++) {
        if (i == size-1){
            receive_c[i] = neigh * end;
        }
        else {
            receive_c[i] = neigh *division;
        }
        receive_d[i] = receive_d[i-1] + neigh* division;
    }

    if (rank == size-1){
        division=end;
    }

    MPI_Gatherv(partialNeighbors, division*neigh, MPI_INT, totalNeighbors, receive_c, receive_d,MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(partialDistances, division*neigh, MPI_DOUBLE, totalDistances, receive_c, receive_d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank==0){
        for (int i = 0; i<querySet.n_cols; i++){
            for (int j = 0; j< neigh; j++){
                localNeighbors.col(i)(j) = totalNeighbors[i*neigh + j];
                localDistances.col(i)(j) = totalDistances[i*neigh + j];
            }
        }

        delete[] totalNeighbors;
        delete[] totalDistances;
    }

    delete[] partialNeighbors;
    delete[] partialDistances;

}

void ParallelNeighborSearch::nearestNeighborSearch(arma::Mat<size_t> &neighbors, arma::mat &distances, int neigh){
    rank = MPI::COMM_WORLD.Get_rank();
    size = MPI::COMM_WORLD.Get_size();
    arma::Mat<size_t> partialNeighbors;
    arma::mat partialDistances;
    arma::Mat<size_t> localNeighbors;
    arma::mat localDistances;

    if (dType == QUERYSET_DECOMPOSITION){
        arma::mat newQueries;
        domainDecomposition(querySet,rank, size, newQueries);
        NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(referenceSet, NAIVE_MODE);

        nn.Search(newQueries, neigh, partialNeighbors, partialDistances);

        localNeighbors.set_size(neigh, querySet.n_cols);
        localDistances.set_size(neigh, querySet.n_cols);

        querySetPostProcessing(partialNeighbors, partialDistances, localNeighbors, localDistances, rank, neigh, newQueries.n_cols);

    } else{
        arma::mat newReferences;
        domainDecomposition(referenceSet, rank, size, newReferences);
        NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(newReferences, NAIVE_MODE);

        nn.Search(querySet, neigh, localNeighbors, localDistances);

        referenceSetPostProcessing(localNeighbors, localDistances, rank, size, NEAREST_NEIGHBOR, neigh);
    }
    if (rank == 0) {
        neighbors = localNeighbors;
        distances = localDistances;
    }
}

void ParallelNeighborSearch::furthestNeighborSearch(arma::Mat<size_t> &neighbors, arma::mat &distances, int neigh){
    rank = MPI::COMM_WORLD.Get_rank();
    size = MPI::COMM_WORLD.Get_size();
    arma::Mat<size_t> partialNeighbors;
    arma::mat partialDistances;
    arma::Mat<size_t> localNeighbors;
    arma::mat localDistances;

    if (dType == QUERYSET_DECOMPOSITION){
        arma::mat newQueries;
        domainDecomposition(querySet,rank, size, newQueries);
        NeighborSearch<FurthestNeighborSort, ManhattanDistance> nn(referenceSet, NAIVE_MODE);

        nn.Search(newQueries, neigh, partialNeighbors, partialDistances);

        localNeighbors.set_size(neigh, querySet.n_cols);
        localDistances.set_size(neigh, querySet.n_cols);

        querySetPostProcessing(partialNeighbors, partialDistances, localNeighbors, localDistances, rank, neigh, newQueries.n_cols);

    } else{
        arma::mat newReferences;
        domainDecomposition(referenceSet, rank, size, newReferences);
        NeighborSearch<FurthestNeighborSort, ManhattanDistance> nn(newReferences, NAIVE_MODE);

        nn.Search(querySet, neigh, localNeighbors, localDistances);

        referenceSetPostProcessing(localNeighbors, localDistances, rank, size, FURTHEST_NEIGHBOR, neigh);
    }
    if (rank == 0) {
        neighbors = localNeighbors;
        distances = localDistances;
    }
}

int main(int argc, char * argv[])  {
    bool _quer, _ref, _neigh, _dtype,_method = false;
    char * ref, * quer;
    bool method = FURTHEST_NEIGHBOR;
    string exitFile = "results.txt";
    int neigh, rank;
    ofstream resultsfile;
    DecompositionType dtype;
    arma::Mat<size_t> neighbors;
    arma::mat distances;


    for (int i=0;i<argc;i++){
        if (strcmp("-q", argv[i]) == 0){
            quer = argv[i+1];
            i++;
            _quer = true;
        }
        if (strcmp("-r", argv[i]) == 0){
            ref = argv[i+1];
            i++;
            _ref = true;
        }
        if (strcmp("-n", argv[i]) == 0){
            neigh = stoi(argv[i+1]);
            i++;
            _neigh = true;
        }
        if (strcmp("-d", argv[i]) == 0) {
            if ((strcmp("query", argv[i+1])) == 0){
                dtype = QUERYSET_DECOMPOSITION;
            }
            else {
                if((strcmp("reference", argv[i+1])) == 0){
                    dtype = REFERENCESET_DECOMPOSITION;
                }
                else {
                    cout << "Please insert either \'reference\' or \'query\' as the decomposition type" << endl;
                    exit(-1);
                }
            }
            i++;
            _dtype = true;
        }
        if (strcmp("-o", argv[i]) == 0) {
            exitFile = argv[i+1];
            i++;
        }
        if(strcmp("-m", argv[i]) == 0){
           if ((strcmp("fns", argv[i+1])) == 0){
                method = FURTHEST_NEIGHBOR;
            }
            else {
                if((strcmp("nns", argv[i+1])) == 0){
                    method = NEAREST_NEIGHBOR;
                }
                else {
                    cout << "Please insert either \'fns\' or \'nns\' as the method to use" << endl;
                    exit(-1);
                }
            }
            i++;
            _method = true; 
        }
    }

    if (!(_dtype && _quer && _ref && _neigh)) {
        cout << "Please provide the querySet, the referenceSet, the number of neighbors to search and the decompositionType" <<endl;
    }

    if (!(_method)) {
        cout << "No method  provided, Furthest Neighbor Search will be executed by  default." << endl;
    }

    MPI::Init();

    ParallelNeighborSearch parallelsearch(quer, ref, dtype);

    rank = MPI::COMM_WORLD.Get_rank();

    if (method == FURTHEST_NEIGHBOR) {
        parallelsearch.furthestNeighborSearch(neighbors, distances, neigh);

        if (rank ==0) {
            resultsfile.open(exitFile, std::ofstream::out);
            for (size_t j=0; j< neighbors.n_cols;j++){
                for (size_t i = 0; i<neigh;i++){
                    resultsfile << i << "º Furthest neighbor of point " << j << " is point " << neighbors.col(j)(i)<< " and the distance is " << distances.col(j)(i) << "." << endl;
                }
            }
            resultsfile.close();
        }
    }
    else {
        parallelsearch.nearestNeighborSearch(neighbors, distances, neigh);

        if (rank ==0) {
            resultsfile.open(exitFile, std::ofstream::out);
            for (size_t j=0; j< neighbors.n_cols;j++){
                for (size_t i = 0; i<neigh;i++){
                    resultsfile << i << "º Nearest neighbor of point " << j << " is point " << neighbors.col(j)(i)<< " and the distance is " << distances.col(j)(i) << "." << endl;
                }
            }
            resultsfile.close();
        }
    }
    

    MPI::Finalize();
}
