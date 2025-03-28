#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <vector>
#include "k_means.h"

// g++-14 -std=c++11 -O3 -fopenmp kmeans.cpp -o kmeans

void load_MNIST(const char* images_file, const char* labels_file, std::vector< std::vector<float> > &images,
std::vector<int> &labels ) {
    int rows = 70000, cols=784;

    std::ifstream file(images_file);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    // resize matrix
    images.resize(rows);
    for (auto &i : images)
        i.resize(cols);
    // Read the matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file >> images[i][j];
        }
    }
    file.close();

    std::ifstream file2(labels_file);
    if (!file2) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    // resize matrix
    labels.resize(rows);
    // Read the matrix elements
    for (int i = 0; i < rows; i++)
        file2 >> labels[i];
    file2.close();
}

constexpr int MAX_N_THREADS = 12;

int findBestK(const std::vector<std::vector<float>>& images, const std::vector<int>& labels,
std::vector<double>& times, std::vector<double>& inertias, std::vector<double>& nmis){
    for(int i = 2; i <= 10; i++){
        std::cout << "\nk = " << i << std::endl;
        auto km = k_means(images, labels, MAX_N_THREADS, i, 70000, 300);
        const auto time = omp_get_wtime();
        auto errors = km.fit(images, 0.0001);
        times[i] = omp_get_wtime() - time;
        inertias[i] = errors[0];
        nmis[i] = errors[1];
        std::cout << "inertia value: " << errors[0] << std::endl
        << "nmi value: " << errors[1] << std::endl << std::endl;
    }
    const auto bestNmi = std::max_element(nmis.begin(), nmis.end());
    const auto bestNmiIdx = std::distance(nmis.begin(), bestNmi);
    std::cout << "Best nmi value: " << *bestNmi << " at k = " << bestNmiIdx << std::endl;
    return bestNmiIdx;
}

int findBestBatchSize(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k,
std::vector<double>& times, std::vector<double>& inertias, std::vector<double>& nmis){
    for(auto i = 2500; i <= 70000; i += 2500){
        std::cout << "\nbatchSize = " << i << std::endl;
        auto km = k_means(images, labels, MAX_N_THREADS, k, i, 300);
        const auto time = omp_get_wtime();
        auto errors = km.fit(images, 0.0001);
        times[i] = omp_get_wtime() - time;
        inertias[i] = errors[0];
        nmis[i] = errors[1];
        std::cout << "inertia value: " << errors[0] << std::endl
        << "nmi value: " << errors[1] << std::endl << std::endl;
    }
    // best inertia value and index in array
    const auto bestInertia = std::min_element(inertias.begin(), inertias.end());
    const auto bestInertiaIdx = std::distance(inertias.begin(), bestInertia);
    // best nmi value and index in array
    const auto bestNmi = std::max_element(nmis.begin(), nmis.end());
    const auto bestNmiIdx = std::distance(nmis.begin(), bestNmi);
    std::cout << "Best inertia value: " << *bestInertia << " at batchSize = " << bestInertiaIdx << std::endl
    << "Best nmi value: " << *bestNmi << " at batchSize = " << bestNmiIdx << std::endl;
    return bestInertiaIdx;
}

void execute(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k, const int batchSize,
std::vector<double>& times, std::vector<double>& speedups){
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "\n# threads = " << i << std::endl;
        auto km = k_means(images, labels, i, k, batchSize, 300);
        const auto time = omp_get_wtime();
        auto errors = km.fit(images, 0.0001);
        times[i-1] = omp_get_wtime() - time;
        speedups[i-1] = times[0] / times[i-1];
        std::cout << "inertia value: " << errors[0] << std::endl
        << "nmi value: " << errors[1] << std::endl << std::endl;
    }
}

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    load_MNIST("../data/mnist-images.txt", "../data/mnist-labels.txt", images, labels);

    // test dataset loading
    std::cout << "No. Images: " << images.size() << std::endl;
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++)
            std::cout<<images[0][i*28+j] << " ";
        std::cout << std::endl;
    }
    std::cout << "Image is " << labels[0] << std::endl;

    int k = 10;
    int batchSize = 2500;
    std::vector<double> times(MAX_N_THREADS, std::numeric_limits<double>::max());
    std::vector<double> speedups(MAX_N_THREADS, 0);
    std::vector<double> inertias(10, std::numeric_limits<double>::max());
    std::vector<double> nmis(10, 0.0);
    // k = 8 is the best
    std::cout << "Finding best k..." << std::endl;
    //k = findBestK(images, labels, times, inertias, nmis);
    // best batchSize = ???
    std::cout << "Finding best batchSize..." << std::endl;
    batchSize = findBestBatchSize(images, labels, k, times, inertias, nmis);
    std::cout << "Fitting k_means..." << std::endl;
    execute(images, labels, k, batchSize, times, speedups);
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "# threads: " << i << std::endl
        << "Time: " << times[i-1] << " Speedup: " << speedups[i-1] << std::endl;
    }

    return 0;
}