#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <unordered_map>

class k_means{
public:
    k_means(std::vector<std::vector<float>>&& data, std::vector<int>&& labels, int k, int batchSize, int maxIter); // constructor
    void fit(float tol);

private:
    std::vector<std::vector<float>> dataset; // data points
    std::vector<int> labels; // true label assignments
    std::unordered_map<int, int> clusters; // fitted cluster assignments (key: cluster idx, value: data point idx)
    int k; // number of clusters
    int batchSize; // number of samples to use in each iteration
    int maxIter; // maximum number of iterations
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
    void scanAssign();
    float inertiaError();
    //TODO: add nmi error method
};

#endif //K_MEANS_H
