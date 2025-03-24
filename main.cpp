#include <fstream>
#include <iostream>
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

    // test k_means with n samples
    std::vector<std::vector<float>> data;
    std::vector<int> labels2;
    for (int i=0; i<10000; i++) {
        data.push_back(images[i]);
        labels2.push_back(labels[i]);
    }
    k_means km((std::move(data)), labels2, 10, 1000, 500);
    auto time = omp_get_wtime();
    auto [fst, snd] = km.fit(1, 0.1);
    time = omp_get_wtime() - time;
    std::cout << "Time: " << time << std::endl
    << "inertia value: " << fst << std::endl
    << "nmi value: " << snd << std::endl;
    return 0;
}