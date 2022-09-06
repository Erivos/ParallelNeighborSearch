#ifndef NEIGHBORSEARCH_H
#define NEIGHBORSEARCH_H

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include "mpi.h"

using namespace std;
using namespace std::chrono;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::metric;


typedef pair<int, double> pointDistance;

enum DecompositionType {
    QUERYSET_DECOMPOSITION,
    REFERENCESET_DECOMPOSITION
};


struct pointDistanceFurthestComparator {
    constexpr bool operator()(
        pair<int, double> const& a,
        pair<int, double> const& b)
        const noexcept
    {
        return a.second > b.second;
    }
};

struct pointDistanceNearestComparator {
    constexpr bool operator()(
        pair<int, double> const& a,
        pair<int, double> const& b)
        const noexcept
    {
        return a.second < b.second;
    }
};

class ParallelNeighborSearch{
    private:
        int rank;
        int size;
        arma::mat querySet;
        arma::mat referenceSet;
        DecompositionType dType;

        void getPartialResults(double partialDistances[], int partialNeighbors[], arma::mat &distances, arma::Mat<size_t> &neighbors, int neigh, MPI_Comm comm, bool nearest = false);
        arma::mat readCSV(const std::string &filename, const std::string &delimeter = ",");
        void serializePartialResults(arma::Mat<size_t> &neighbors, arma::mat &distances, int partialNeighbors[], double partialDistances[],  int neigh, int rank, int size);

        void domainDecomposition(arma::mat &set, int rank, int size, arma::mat &setProc);

        void referenceSetPostProcessing(arma::Mat<size_t> &neighbors, arma::mat &distances,  int rank, int size, bool nearest, int neigh);
        void querySetPostProcessing(arma::Mat<size_t> &neighbors, arma::mat &distances, arma::Mat<size_t> &localNeighbors, arma::mat &localDistances, int rank, int neigh, int ProcNCols);
    public:
        ParallelNeighborSearch(std::string querySetFilename, std::string referenceSetFilename, DecompositionType initialDType);

        void nearestNeighborSearch(arma::Mat<size_t> &neighbors, arma::mat &distances, int neigh);
        void furthestNeighborSearch(arma::Mat<size_t> &neighbors, arma::mat &distances, int neigh);

        void setDecompositionType(DecompositionType newDType);
        };


#endif
