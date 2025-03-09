#include "k_means.h"
#include <random>

k_means::k_means(std::vector<std::vector<float>>&& data, std::vector<int>&& labels, const int k, const int batchSize,
const int maxIter) : dataset(std::move(data)), labels(std::move(labels)), k(k), batchSize(batchSize), maxIter(maxIter) {
    assignments.resize(labels.size());
    centroids.resize(k);
    for(auto& a : assignments){ // initialize the assignments to -1 (unassigned)
        a = -1;
    }
    for( auto& c : centroids){ // initialize the centroids to random values
        c.resize(784);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for(auto i = 0; i < 784; i++){
            c[i] = dis(gen);
        }
    }
}

float k_means::euclideanDistance(const std::vector<float>& x, const int c_idx) const{
    float sum = 0.0;
    for(auto i = 0; i < x.size(); i++){ // calculate the euclidean distance between a data point and a centroid
        sum += std::pow(x[i] - centroids[c_idx][i], 2); // sum of squared differences
    }
    return std::sqrt(sum);
}

std::vector<std::vector<float>> k_means::sampleData() const{
    std::vector<std::vector<float>> batch;
    std::vector<int> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0); // fill the vector with 0, 1, 2, ..., dataset.size()-1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen); // shuffle the indices
    batch.reserve(batchSize);
for(auto i = 0; i < batchSize; i++){ // sample batchSize shuffled data points
        batch.push_back(dataset[indices[i]]);
    }
    return batch;
}

int k_means::findCentroidIdx(const std::vector<float>& x) const{
    int idx = 0;
    for(auto i = 1; i < k; i++){ // find the closest centroid idx for a data point using euclidean distance
        if(euclideanDistance(x, i) < euclideanDistance(x, idx))
            idx = i;
    }
    return idx;
}

void k_means::updateCentroid(const std::vector<float>& x, std::vector<int>& counts, const int idx){
    counts[idx] += 1; // add one to the count of data points assigned to the centroid
    const float ln = 1.0 / counts[idx]; // learning rate decays with the number of data points assigned to the centroid
    for(auto i = 0; i < k; i++){ // update the centroid based on a data point
        centroids[idx][i] = (1 - ln) * centroids[idx][i] + ln * x[i]; // lower ln, less weight to the new data point
    }
}

void k_means::assignData(){
    for(auto i = 0; i < dataset.size(); i++){ // assign data points to the closest centroid
        assignments[i] = findCentroidIdx(dataset[i]);
    }
}
