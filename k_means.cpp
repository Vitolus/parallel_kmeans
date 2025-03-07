#include "k_means.h"

k_means::k_means(const int k, const int batchSize, const int maxIter) : k(k), batchSize(batchSize), maxIter(maxIter) {
    centroids.resize(k);
}
