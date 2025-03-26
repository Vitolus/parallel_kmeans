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

int main() {
    std::vector< std::vector<float> > images;
    std::vector<int> labels;
    load_MNIST("../data/mnist-images.txt", "../data/mnist-labels.txt", images, labels);

    // test dataset loading
    /*
    std::cout << "No. Images: " << images.size() << std::endl;
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++)
            std::cout<<images[0][i*28+j] << " ";
        std::cout << std::endl;
    }
    std::cout << "Image is " << labels[0] << std::endl;
    */


    std::cout << "Fitting k_means..." << std::endl;
    std::vector<float> times(20, std::numeric_limits<float>::max());
    std::vector<float> speedups(20, 0);
    std::vector<float> inertias(10, std::numeric_limits<float>::max());
    std::vector<float> nmis(10, 0.0);
    for(int i = 2; i < 10; i++){
        auto km = k_means(std::vector<std::vector<float>>(images), labels, 12, i, 70000, 300);
        const auto time = omp_get_wtime();
        auto [fst, snd] = km.fit(0.0001);
        times[i] = omp_get_wtime() - time;
        inertias[i] = fst;
        nmis[i] = snd;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl;
    }
    // best inertia value and index in array
    const auto best_inertia = std::min_element(inertias.begin(), inertias.end());
    const auto best_inertia_idx = std::distance(inertias.begin(), best_inertia);
    // best nmi value and index in array
    const auto best_nmi = std::max_element(nmis.begin(), nmis.end());
    const auto best_nmi_idx = std::distance(nmis.begin(), best_nmi);
    std::cout << "Best inertia value: " << *best_inertia << " at k = " << best_inertia_idx << std::endl
    << "Best nmi value: " << *best_nmi << " at k = " << best_nmi_idx << std::endl;
    for(int i = 13; i <= 12; i++){
        auto km = k_means(std::vector<std::vector<float>>(images), labels, i, 10, 70000, 300);
        const auto time = omp_get_wtime();
        auto [fst, snd] = km.fit(0.0001);
        times[i-1] = omp_get_wtime() - time;
        speedups[i-1] = times[0] / times[i-1];
        std::cout << "# threads: " << i << std::endl
        << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl;
    }
    for(int i = 1; i <= 20; i++){
        std::cout << "# threads: " << i << std::endl
        << "Time: " << times[i-1] << " Speedup: " << speedups[i-1] << std::endl;
    }

    return 0;
}