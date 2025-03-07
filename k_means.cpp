#include "k_means.h"
#include <random>

k_means::k_means(const int k, const int batchSize, const int maxIter) : k(k), batchSize(batchSize), maxIter(maxIter) {
    centroids.resize(k);
    for( auto &c : centroids){
        c.resize(784);
        // initialize the centroids to random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for(auto i = 0; i < 784; i++){
            c[i] = dis(gen);
        }
    }
}

float k_means::euclideanDistance(const std::vector<float> &x, const int c_idx) const{
    float sum = 0.0;
    for(auto i = 0; i < x.size(); i++){
        sum += std::pow(x[i] - centroids[c_idx][i], 2);
    }
    return std::sqrt(sum);
}

std::vector<std::vector<float>> k_means::sampleData(const std::vector<std::vector<float>> &dataset){
    std::vector<std::vector<float>> batch;
    std::vector<auto> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    for(auto i = 0; i < batchSize; i++){
        batch.push_back(dataset[indices[i]]);
    }
    return batch;
}

int k_means::findCentroidIdx(const std::vector<float>& x) const{
    int idx = 0;
    for(auto i = 1; i < k; i++){
        if(euclideanDistance(x, i) < euclideanDistance(x, idx))
            idx = i;
    }
    return idx;
}

void k_means::updateCentroid(const std::vector<float>& x, std::vector<int>& counts, const int idx){
    counts[idx] += 1;
    const float ln = 1.0 / counts[idx];
    for(auto i = 0; i < k; i++){
        centroids[idx][i] = (1 - ln) * centroids[idx][i] + ln * x[i];
    }
}
