#include "kmeans_euclidean.h"
#include <cmath>

float kmeans_euclidean::distance(const std::vector<float> &x, const int c_idx) {
    float sum = 0.0;
    for(auto i = 0; i < x.size(); i++){
    sum += std::pow(x[i] - centroids[c_idx][i], 2);
    }
    return std::sqrt(sum);
}
