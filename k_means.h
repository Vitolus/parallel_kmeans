#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <unordered_map>

class k_means{
public:
    k_means(std::vector<std::vector<float>>&& data, const std::vector<int>& labels, int k, int batchSize, int maxIter); // constructor
    void fit(float tol);

private:
    int k; // number of clusters
    int batchSize; // number of samples to use in each iteration
    int maxIter; // maximum number of iterations
    std::vector<std::vector<float>> dataset; // data points
    std::vector<std::vector<int>> labelClusters;
    std::vector<std::vector<int>> clusters; // fitted cluster assignments (key: cluster idx, value: data point idx)
    std::vector<std::vector<float>> centroids; // cluster centers

    // euclidean distance between a data point and a cluster center
    [[nodiscard]] float euclideanDistance(const std::vector<float>& x, int c_idx) const;
    // sample a batch of data points
    [[nodiscard]] std::vector<std::vector<float>> sampleData() const;
    // find the closest centroid idx for a data point
    [[nodiscard]] int findCentroidIdx(const std::vector<float>& x) const;
    // update the centroid based on a data point
    void updateCentroid(const std::vector<float>& x, std::vector<int>& counts, int idx);
    // assign data points to the closest centroid
    void scanAssign(const std::vector<std::vector<float>>& batch);
    // sum of squared distances of samples to their closest cluster center
    float inertiaError(const std::vector<std::vector<float>>& batch);
    // normalized mutual information
    float nmiError();
};

#endif //K_MEANS_H
