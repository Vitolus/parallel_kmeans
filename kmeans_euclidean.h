#ifndef KMEANS_EUCLIDEAN_H
#define KMEANS_EUCLIDEAN_H
#include "k_means.h"

class kmeans_euclidean : public k_means{
public:
    using k_means::k_means;
    float distance(const std::vector<float>& a, const std::vector<float>& b) override;
};

#endif //KMEANS_EUCLIDEAN_H
