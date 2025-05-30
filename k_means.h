#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <omp.h>

class k_means{
    int n_threads;
    int k; // number of clusters
    int batchSize; // number of samples to use in each iteration
    int maxIter; // maximum number of iterations
    std::vector<std::vector<int>> labelClusters;
    std::vector<std::vector<int>> clusters; // fitted cluster assignments (idx: cluster idx, value: array of data point idx)
    std::vector<std::vector<float>> centroids; // cluster centers

    // Euclidean distance between a data point and a cluster center
    [[nodiscard]] double squaredEuclideanDistance(const std::vector<float>& x, int c_idx) const;
    // compute the difference in change of the centroids
    [[nodiscard]] double deltaChange(double& prevChange, std::vector<std::vector<float>>& prevCentroids) const;
    // sample a batch of data points
    [[nodiscard]] std::vector<std::vector<float>> sampleData(const std::vector<std::vector<float>>& dataset) const;
    // find the closest centroid idx for a data point
    [[nodiscard]] int findCentroidIdx(const std::vector<float>& x) const;
    // assign data points to the closest centroid
    void scanAssign(const std::vector<std::vector<float>>& batch);
    // sum of squared distances of samples to their closest cluster center
    double inertiaError(const std::vector<std::vector<float>>& dataset);
    // normalized mutual information
    double nmiError(int size);

public:
    k_means(const std::vector<std::vector<float>>& dataset, const std::vector<int>& labels, int n_threads, int k, int batchSize, int maxIter);
    std::pair<double, double> fit(const std::vector<std::vector<float>>& dataset, double tol);
};

#endif //K_MEANS_H
