#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>

class k_means{
public:
    k_means(int k, int batchSize, int maxIter);
    virtual ~k_means() = default;
    virtual float distance(const std::vector<float>& a, const std::vector<float>& b) = 0;

protected:
    int k;
    int batchSize;
    int maxIter;
    std::vector<std::pair<float, int>> centroids; // (ci, vi)
};

#endif //K_MEANS_H
