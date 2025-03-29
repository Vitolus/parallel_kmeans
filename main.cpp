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

int findBestK(const std::vector<std::vector<float>>& images, const std::vector<int>& labels){
    int maxNmiIdx = 0;
    std::vector<double> nmis(9);
    int j = 0;
    for(int i = 2; i <= 10; i++){
        std::cout << "\nk = " << i << std::endl;
        auto* km = new k_means(images, labels, MAX_N_THREADS, i, 70000, 300);
        auto [fst, snd] = km->fit(images, 0.0001);
        delete km;
        km = nullptr;
        nmis[j] = snd;
        if(nmis[maxNmiIdx] < snd) maxNmiIdx = j;
        j++;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl << std::endl;
    }
    std::cout << "Best nmi value: " << nmis[maxNmiIdx] << "at k = " << maxNmiIdx + 2 << std::endl;
    return maxNmiIdx;
}

int findBestBatchSize(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k){
    int minInertiaIdx = 0;
    int maxNmiIdx = 0;
    std::vector<double> inertias(28);
    std::vector<double> nmis(28);
    int j = 0;
    for(int i = 2500; i <= 70000; i += 2500){
        std::cout << "\nbatchSize = " << i << std::endl;
        auto* km = new k_means(images, labels, MAX_N_THREADS, k, i, 300);
        auto [fst, snd] = km->fit(images, 0.0001);
        delete km;
        km = nullptr;
        inertias[j] = fst;
        nmis[j] = snd;
        if(nmis[minInertiaIdx] > fst) minInertiaIdx = j;
        if(inertias[maxNmiIdx] < snd) maxNmiIdx = j;
        j++;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl << std::endl;
    }
    std::cout << "Best inertia value: " << inertias[minInertiaIdx] << "at batchSize = " << (minInertiaIdx + 1) * 2500 << std::endl
    << "Best nmi value: " << nmis[maxNmiIdx] << "at batchSize = " << (maxNmiIdx + 1) * 2500 << std::endl;
    return minInertiaIdx;
}

void execute(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k, const int batchSize,
std::vector<double>& times, std::vector<double>& speedups){
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "\n# threads = " << i << std::endl;
        auto* km = new k_means(images, labels, i, k, batchSize, 300);
        const auto time = omp_get_wtime();
        auto [fst, snd] = km->fit(images, 0.0001);
        times[i-1] = omp_get_wtime() - time;
        delete km;
        km = nullptr;
        speedups[i-1] = times[0] / times[i-1];
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl << std::endl;
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

    int k = 9;
    int batchSize = 70000;
    std::vector<double> times(MAX_N_THREADS, std::numeric_limits<double>::max());
    std::vector<double> speedups(MAX_N_THREADS);
    // k = 8 is the best
    std::cout << "\nFinding best k..." << std::endl;
    k = findBestK(images, labels);
    // best batchSize = ???
    std::cout << "\nFinding best batchSize..." << std::endl;
    batchSize = findBestBatchSize(images, labels, k);

    for(auto i = 0; i < 5; i++){
        std::cout << "\nExecution " << i << std::endl;
        auto* km = new k_means(images, labels, MAX_N_THREADS, k, batchSize, 300);
        const auto time = omp_get_wtime();
        auto [fst, snd] = km->fit(images, 0.0001);
        times[0] = omp_get_wtime() - time;
        delete km;
        km = nullptr;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl << std::endl;
    }

    std::cout << "\nFitting k_means..." << std::endl;
    execute(images, labels, k, batchSize, times, speedups);
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "# threads: " << i << std::endl
        << "Time: " << times[i-1] << " Speedup: " << speedups[i-1] << std::endl;
    }

    return 0;
}