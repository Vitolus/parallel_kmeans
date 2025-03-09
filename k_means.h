#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>

class k_means{
public:
    k_means(std::vector<std::vector<float>>&& data, std::vector<int>&& labels, int k, int batchSize, int maxIter); // constructor

private:
    std::vector<std::vector<float>> dataset; // data points
    std::vector<int> labels; // true label assignments
    std::vector<int> assignments; // fitted assignments
    int k; // number of clusters
    int batchSize; // number of samples to use in each iteration
    int maxIter; // maximum number of iterations
    std::vector<std::vector<float>> centroids; // cluster centers
    //TODO:2 implement fit method

    // euclidean distance between a data point and a cluster center
    [[nodiscard]] float euclideanDistance(const std::vector<float>& x, int c_idx) const;
    // sample a batch of data points
    std::vector<std::vector<float>> sampleData();
    // find the closest centroid idx for a data point
    [[nodiscard]] int findCentroidIdx(const std::vector<float>& x) const;
    // update the centroid based on a data point
    void updateCentroid(const std::vector<float>& x, std::vector<int>& counts, int idx);
    // assign data points to the closest centroid
    void assignData();
};

#endif //K_MEANS_H
