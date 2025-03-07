#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>

class k_means{
public:
    k_means(int k, int batchSize, int maxIter); // constructor
    virtual ~k_means() = default; // destructor

protected:
    int k; // number of clusters
    int batchSize; // number of samples to use in each iteration
    int maxIter; // maximum number of iterations
    std::vector<std::vector<float>> centroids; // cluster centers

    // distance between a data point and a cluster center
    virtual float distance(const std::vector<float>& a, int c_idx) = 0;
    // sample a batch of data points
    std::vector<std::vector<float>> sampleData(const std::vector<std::vector<float>>& dataset);
    // find the closest centroid idx for a data point
    int findCentroidIdx(const std::vector<float>& x);
};

#endif //K_MEANS_H
