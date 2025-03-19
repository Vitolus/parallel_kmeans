#include "k_means.h"
#include <random>

k_means::k_means(std::vector<std::vector<float>>&& data, const std::vector<int>& labels, const int k, const int batchSize,
const int maxIter) : k(k), batchSize(batchSize), maxIter(maxIter), dataset(std::move(data)) {
    centroids.resize(k);
    for( auto& centroid : centroids){ // initialize the centroids to random values
        centroid.resize(784);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for(auto i = 0; i < 784; i++){
            centroid[i] = dis(gen);
        }
    }
    clusters.resize(k);
    labelClusters.resize(10);
    for(auto i = 0; i < labels.size(); i++){ // true cluster assignments
        labelClusters[labels[i]].push_back(i);
    }
    for(auto& cluster : labelClusters){
        cluster.shrink_to_fit();
    }
}

float k_means::fit(const float tol){
    std::vector<int> counts(k, 0); // count the number of data points assigned to each centroid
    float deltaError = std::numeric_limits<float>::max();
    float prevError = 0.0;
    for(auto i = 0; deltaError > tol && i < maxIter; i++){
        std::vector<std::vector<float>> batch = sampleData(); // sample a batch of data points
        for(const auto& x : batch){
            const int idx = findCentroidIdx(x); // find the closest centroid idx for a data point
            updateCentroid(x, counts, idx); // update the centroid based on a data point
        }
        scanAssign(batch); // assign data points to the closest centroid
        const float currError = inertiaError(batch); // calculate the inertia error
        deltaError = std::abs(prevError - currError); // calculate the change in inertia error
        prevError = currError;
    }
    scanAssign(dataset);
    return nmiError();
}

float k_means::euclideanDistance(const std::vector<float>& x, const int c_idx) const{
    float sum = 0.0;
    for(auto i = 0; i < x.size(); i++){ // calculate the Euclidean distance between a data point and a centroid
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
    for(auto i = 1; i < k; i++){ // find the closest centroid idx for a data point using Euclidean distance
        if(euclideanDistance(x, i) < euclideanDistance(x, idx))
            idx = i;
    }
    return idx;
}

void k_means::updateCentroid(const std::vector<float>& x, std::vector<int>& counts, const int idx){
    counts[idx] += 1; // add one to the count of data points assigned to the centroid
    const float lr = 1.0 / counts[idx]; // learning rate decays with the number of data points assigned to the centroid
    for(auto i = 0; i < k; i++){ // update the centroid based on a data point
        centroids[idx][i] = (1 - lr) * centroids[idx][i] + lr * x[i]; // lower ln, less weight to the new data point
    }
}

void k_means::scanAssign(const std::vector<std::vector<float>>& batch){
    clusters.clear();
    for(auto i = 0; i < batch.size(); i++){ // assign data points to the closest centroid
        clusters[findCentroidIdx(batch[i])].push_back(i);
    }
    for(auto& cluster : clusters){
        cluster.shrink_to_fit();
    }
}

float k_means::inertiaError(const std::vector<std::vector<float>>& batch){
    float inertia = 0.0;
    for(auto i = 0; i < k; i++){
        for(const int& p : clusters[i]){ // sum of squared distances of samples to their closest cluster center
            inertia += euclideanDistance(batch[p], i);
        }
    }
    return inertia;
}

float k_means::nmiError(){
    float nmi = 0.0;
    float hentropyClusters = 0.0;
    float hentropyLabels = 0.0;
    float mutualInformation = 0.0;
    for(const auto & cluster : clusters){
        const float ratio = cluster.size() / dataset.size();
        hentropyClusters += ratio * (-1)*std::log2(ratio);
    }
    for(const auto & labelCluster : labelClusters){
        const float ratio = labelCluster.size() / dataset.size();
        hentropyLabels += ratio * (-1)*std::log2(ratio);
    }
    for(auto & cluster : clusters){
        for(auto & labelCluster : labelClusters){
            std::vector<int> intersection;
            std::set_intersection(cluster.begin(), cluster.end(),
            labelCluster.begin(), labelCluster.end(), std::back_inserter(intersection));
            mutualInformation += (intersection.size() / dataset.size()) * std::log2((dataset.size() *
            intersection.size()) / (cluster.size() * labelCluster.size()));
        }
    }
    return mutualInformation / ((hentropyClusters + hentropyLabels) / 2);
}


