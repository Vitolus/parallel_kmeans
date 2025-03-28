#include "k_means.h"
#include <iostream>
#include <random>

k_means::k_means(const std::vector<std::vector<float>>& dataset, const std::vector<int>& labels, const int n_threads, const int k, const int batchSize,
const int maxIter) : n_threads(n_threads), k(k), batchSize(batchSize), maxIter(maxIter) {
    centroids.resize(k);
    clusters.resize(k);
    labelClusters.resize(10);
    for(auto i = 0; i < labels.size(); i++){ // true cluster assignments
        labelClusters[labels[i]].push_back(i);
    }
    for(auto& cluster : labelClusters){
        cluster.shrink_to_fit();
    }
    for(auto i = 0; i < k; i++){
        auto& centroid = centroids[i];
        centroid = dataset[labelClusters[i][0]]; // initialize the centroids to the first data point of each true cluster
    }
}

std::pair<double, double> k_means::fit(const std::vector<std::vector<float>>& dataset, const double tol){
    std::vector<int> counts(k, 0); // count the number of data points assigned to each centroid
    double deltaChange = tol + 1.0;
    double prevChange = 0.0;
    std::vector<std::vector<float>> prevCentroids = centroids;
    auto i = 0;
    for(i = 0; deltaChange > tol && i < maxIter; i++){
        std::vector<std::vector<float>> batch = sampleData(dataset); // sample a batch of data points
        std::vector<int> indices(batchSize);
        #pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
        for(int j = 0; j < batchSize; j++){
            const auto& x = batch[j];
            indices[j] = findCentroidIdx(x); // find the closest centroid idx for a data point
        }
        // no parallelization improvement; give different results because changes order of updates, destroying seeding
        for(int j = 0; j < batchSize; j++){
            const auto& x = batch[j];
            updateCentroid(x, counts, indices[j]); // update the centroid based on a data point
        }
        scanAssign(batch); // assign data points to the closest centroid
        double totalChange = 0.0;
        for(auto j = 0; j < k; j++){
        totalChange += euclideanDistance(prevCentroids[j], j);
        }
        deltaChange = std::abs(totalChange - prevChange);
        prevChange = totalChange;
        prevCentroids = centroids;
        //std::cout << "delta change: " << deltaChange << std::endl;
    }
    scanAssign(dataset);

    // show clusters
    for(auto j=0; j<k; j++) {
        std::cout << "Cluster " << j << " size: " << clusters[j].size()
        << " / " << labelClusters[j].size() << std::endl;
    }
    std::cout << "Number of iterations: " << i << std::endl
    << "delta change: " << deltaChange << std::endl;

    return {inertiaError(dataset), nmiError(dataset.size())};
}

double k_means::euclideanDistance(const std::vector<float>& x, const int c_idx) const{
    double sum = 0.0;
    for(auto i = 0; i < 784; i++){ // calculate the Euclidean distance between a data point and a centroid
        sum += std::pow(x[i] - centroids[c_idx][i], 2); // sum of squared differences
    }
    return std::sqrt(sum);
}

std::vector<std::vector<float>> k_means::sampleData(const std::vector<std::vector<float>>& dataset) const{
    std::vector<std::vector<float>> batch(batchSize);
    std::vector<int> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0); // fill the vector with 0, 1, 2, ..., dataset.size()-1
    std::mt19937 gen(666);
    std::shuffle(indices.begin(), indices.end(), gen);
    for(auto i = 0; i < batchSize; i++){ // sample batchSize shuffled data points
        batch[i] = dataset[indices[i]];
    }
    return batch;
}

int k_means::findCentroidIdx(const std::vector<float>& x) const{
    int idx = 0;
    double minDistance = euclideanDistance(x, 0);
    for(auto i = 1; i < k; i++){ // find the closest centroid idx for a data point using Euclidean distance
        if(const double distance = euclideanDistance(x, i); distance < minDistance) {
                minDistance = distance;
                idx = i;
            }
    }
    return idx;
}

void k_means::updateCentroid(const std::vector<float>& x, std::vector<int>& counts, const int idx){
    counts[idx] += 1; // add one to the count of data points assigned to the centroid
    const double lr = 1.0 / counts[idx]; // learning rate decays with the number of data points assigned to the centroid
    for(auto i = 0; i < 784; i++){ // update the centroid based on a data point
        centroids[idx][i] = (1 - lr) * centroids[idx][i] + lr * x[i]; // lower lr, less weight to the new data point
    }
}

void k_means::scanAssign(const std::vector<std::vector<float>>& batch){
    for(auto& cluster : clusters){
        cluster.clear();
    }
    #pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
    for(auto i = 0; i < batch.size(); i++){ // assign data points to the closest centroid
        int idx = findCentroidIdx(batch[i]);
        #pragma omp critical
        clusters[idx].push_back(i);
    }
    for(auto& cluster : clusters){
        cluster.shrink_to_fit();
    }
}

double k_means::inertiaError(const std::vector<std::vector<float>>& dataset){
    double inertia = 0.0;
    for(auto i = 0; i < k; i++){
        for(const int p : clusters[i]){ // sum of squared distances of samples to their closest cluster center
            inertia += std::pow(euclideanDistance(dataset[p], i), 2);
        }
    }
    return inertia;
}

double k_means::nmiError(const int size){
    double hentropyClusters = 0.0;
    double hentropyLabels = 0.0;
    double mutualInformation = 0.0;
    for(const auto& cluster : clusters){
        const double ratio = static_cast<double>(cluster.size()) / size;
        hentropyClusters += ratio * std::log2(ratio);
    }
    hentropyClusters *= -1;
    for(const auto& labelCluster : labelClusters){
        const double ratio = static_cast<double>(labelCluster.size()) / size;
        hentropyLabels += ratio * std::log2(ratio);
    }
    hentropyLabels *= -1;
    for(auto & cluster : clusters){
        for(auto & labelCluster : labelClusters){
            std::vector<int> intersection;
            std::set_intersection(cluster.begin(), cluster.end(),
            labelCluster.begin(), labelCluster.end(), std::back_inserter(intersection));
            if (intersection.empty()) continue; //by definition, log2(0) = 0
            mutualInformation += (static_cast<double>(intersection.size()) / size) *
            std::log2(static_cast<double>((size * intersection.size())) /
            (cluster.size() * labelCluster.size()));
        }
    }
    return mutualInformation / ((hentropyClusters + hentropyLabels) / 2);
}


