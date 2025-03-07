#include "kmeans_euclidean.h"
#include <cmath>

float kmeans_euclidean::distance(const std::vector<float> &a, const std::vector<float> &b){
    float sum = 0.0;
    for(auto i = 0; i < a.size(); i++){
    sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}
