#include "k_means.h"

#include <cassert>
#include <iostream>
#include <random>

k_means::k_means(const std::vector<std::vector<float>>& dataset, const std::vector<int>& labels, const int n_threads, const int k, const int batchSize,
const int maxIter) : n_threads(n_threads), k(k), batchSize(batchSize), maxIter(maxIter) {
    centroids.resize(k);
    clusters.resize(k);
    labelClusters.resize(10);
    for(size_t i = 0; i < labels.size(); i++){ // true cluster assignments
        labelClusters[labels[i]].push_back(i);
    }
    for(auto& cluster : labelClusters){
        cluster.shrink_to_fit();
    }
    for(size_t i = 0; i < k; i++){
        auto& centroid = centroids[i];
        centroid = dataset[labelClusters[i][0]]; // initialize the centroids to the first data point of each true cluster
    }
}

std::pair<double, double> k_means::fit(const std::vector<std::vector<float>>& dataset, const double tol){
    std::vector<int> counts(k); // count the number of data points assigned to each centroid
    double delta = tol + 1.0;
    double prevChange = 0.0;
    std::vector<std::vector<float>> prevCentroids = centroids;
    std::vector<omp_lock_t> locks(k);
    for(size_t i = 0; i < k; i++){
        omp_init_lock(&locks[i]);
    }
    size_t i = 0;
    for(i = 0; delta > tol && i < maxIter; i++){
        std::vector<std::vector<float>> batch = sampleData(dataset); // sample a batch of data points
        std::vector<int> indices(batchSize);
        #pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
        for(size_t j = 0; j < batchSize; j++){
            const auto& x = batch[j];
            indices[j] = findCentroidIdx(x); // find the closest centroid idx for a data point
        }
        // no parallelization improvement; give different results because changes order of updates, destroying seeding
        //TODO: do tests with and without parallelization to see if it is worth it
        #pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
        for(size_t j = 0; j < batchSize; j++){
            const auto& x = batch[j];
            int idx = indices[j];
            int curCount;
            #pragma omp atomic capture
            curCount = ++counts[idx];
            const double lr = 1.0 / curCount; // learning rate decays with the number of data points assigned to the centroid
            omp_set_lock(&locks[idx]);
            for(size_t l = 0; l < 784; l++){ // update the centroid based on a data point
                centroids[idx][l] = (1 - lr) * centroids[idx][l] + lr * x[l]; // lower lr, less weight to the new data point
            }
            omp_unset_lock(&locks[idx]);
        }
        //scanAssign(batch); // assign data points to the closest centroid
        delta = deltaChange(prevChange, prevCentroids);
        //std::cout << "delta change: " << deltaChange << std::endl;
    }
    for(size_t i = 0; i < k; i++){
        omp_destroy_lock(&locks[i]);
    }
    scanAssign(dataset);

    // show clusters
    for(size_t j=0; j<k; j++) {
        std::cout << "Cluster " << j << " size: " << clusters[j].size()
        << " / " << labelClusters[j].size() << std::endl;
    }
    std::cout << "Number of iterations: " << i << std::endl
    << "delta change: " << delta << std::endl;

    return {inertiaError(dataset), nmiError(dataset.size())};
}

double k_means::squaredEuclideanDistance(const std::vector<float>& x, const int c_idx) const{
    double sum = 0.0;
    for(size_t i = 0; i < 784; i++){ // calculate the Euclidean distance between a data point and a centroid
        const double diff = x[i] - centroids[c_idx][i];
        sum += diff * diff; // sum of squared differences
    }
    return sum;
}

double k_means::deltaChange(double& prevChange, std::vector<std::vector<float>>& prevCentroids) const{
    double delta = 0.0;
    double totalChange = 0.0;
    for(size_t i = 0; i < k; i++){
        totalChange += std::sqrt(squaredEuclideanDistance(prevCentroids[i], i));
    }
    delta = std::abs(totalChange - prevChange);
    prevChange = totalChange;
    prevCentroids = centroids;
    return delta;
}

std::vector<std::vector<float>> k_means::sampleData(const std::vector<std::vector<float>>& dataset) const{
    std::vector<std::vector<float>> batch(batchSize);
    std::vector<int> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0); // fill the vector with 0, 1, 2, ..., dataset.size()-1
    std::mt19937 gen(666);
    std::shuffle(indices.begin(), indices.end(), gen);
    for(size_t i = 0; i < batchSize; i++){ // sample batchSize shuffled data points
        batch[i] = dataset[indices[i]];
    }
    return batch;
}

int k_means::findCentroidIdx(const std::vector<float>& x) const{
    int idx = 0;
    double minDistance = squaredEuclideanDistance(x, 0);
    for(size_t i = 1; i < k; i++){ // find the closest centroid idx for a data point using Euclidean distance
        if(const double distance = squaredEuclideanDistance(x, i); distance < minDistance) {
                minDistance = distance;
                idx = i;
            }
    }
    return idx;
}

void k_means::scanAssign(const std::vector<std::vector<float>>& batch){
    for(auto& cluster : clusters){
        cluster.clear();
    }
    std::vector<std::vector<std::vector<int>>> localClusters(n_threads, std::vector<std::vector<int>>(k));
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for(size_t i = 0; i < batch.size(); i++){ // assign data points to the closest centroid
            int idx = findCentroidIdx(batch[i]);
            localClusters[tid][idx].push_back(i);
        }
    }
    // merge local clusters
    for(size_t t = 0; t < n_threads; t++){
        for(size_t i = 0; i < k; i++){
            clusters[i].insert(clusters[i].end(), localClusters[t][i].begin(), localClusters[t][i].end());
        }
    }
    for(auto& cluster : clusters){
        cluster.shrink_to_fit();
    }
}

double k_means::inertiaError(const std::vector<std::vector<float>>& dataset){
    double inertia = 0.0;
    for(size_t i = 0; i < k; i++){
        for(const int p : clusters[i]){ // sum of squared distances of samples to their closest cluster center
            inertia += squaredEuclideanDistance(dataset[p], i);
        }
    }
    return inertia;
}

double k_means::nmiError(const int size){
    double hentropyClusters = 0.0;
    double hentropyLabels = 0.0;
    double mutualInformation = 0.0;
    for(auto& cluster : clusters) {
        std::sort(cluster.begin(), cluster.end());
        const double ratio = static_cast<double>(cluster.size()) / size;
        if(ratio == 0) continue; //by definition, log2(0) = 0
        hentropyClusters += ratio * std::log2(ratio);
    }
    hentropyClusters *= -1;
    for(auto& labelCluster : labelClusters) {
        std::sort(labelCluster.begin(), labelCluster.end());
        const double ratio = static_cast<double>(labelCluster.size()) / size;
        hentropyLabels += ratio * std::log2(ratio);
    }
    hentropyLabels *= -1;
    for(auto & cluster : clusters){
        for(auto & labelCluster : labelClusters){
            std::vector<int> intersection;
            std::set_intersection(cluster.begin(), cluster.end(),
            labelCluster.begin(), labelCluster.end(), std::back_inserter(intersection));
            if(intersection.empty()) continue; //by definition, log2(0) = 0
            mutualInformation += (static_cast<double>(intersection.size()) / size) *
            std::log2((static_cast<double>(size * intersection.size())) /
            (cluster.size() * labelCluster.size()));
        }
    }
    return mutualInformation / ((hentropyClusters + hentropyLabels) / 2);
}


