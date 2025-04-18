#include <fstream>
#include <iostream>
#include <vector>
#include "k_means.h"

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

void writeCSV(std::ofstream& file, const std::vector<double> &times, const std::vector<double> &speedups){
    file << "n threads,time,speedup" << std::endl;
    for(auto i = 1; i <= times.size(); ++i){
        file << i << "," << times[i-1] << "," << speedups[i-1] << std::endl;
    }
    file.close();
}

int MAX_N_THREADS = 1;
constexpr int MAX_ITER = 500;
constexpr float TOL = 1e-4;

int findBestK(const std::vector<std::vector<float>>& images, const std::vector<int>& labels){
    int maxNmiIdx = 0;
    std::vector<double> nmis(9);
    int j = 0;
    for(int i = 2; i <= 10; i++){
        std::cout << "\nk = " << i << std::endl;
        auto* km = new k_means(images, labels, MAX_N_THREADS, i, 70000, MAX_ITER);
        auto [fst, snd] = km->fit(images, TOL);
        delete km;
        km = nullptr;
        nmis[j] = snd;
        if(nmis[maxNmiIdx] < snd) maxNmiIdx = j;
        j++;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl << std::endl;
    }
    std::cout << "Best nmi value: " << nmis[maxNmiIdx] << "at k = " << maxNmiIdx + 2 << std::endl;
    return maxNmiIdx + 2;
}

int findBestBatchSize(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k){
    int minInertiaIdx = 0;
    int maxNmiIdx = 0;
    std::vector<double> inertias(28);
    std::vector<double> nmis(28);
    int j = 0;
    for(int i = 2500; i <= 70000; i += 2500){
        std::cout << "\nbatchSize = " << i << std::endl;
        auto* km = new k_means(images, labels, MAX_N_THREADS, k, i, MAX_ITER);
        auto [fst, snd] = km->fit(images, TOL);
        delete km;
        km = nullptr;
        inertias[j] = fst;
        nmis[j] = snd;
        if(inertias[minInertiaIdx] > fst) minInertiaIdx = j;
        if(nmis[maxNmiIdx] < snd) maxNmiIdx = j;
        j++;
        std::cout << "inertia value: " << fst << std::endl
        << "nmi value: " << snd << std::endl;
    }
    std::cout << "Best inertia value: " << inertias[minInertiaIdx] << " at batchSize = " << (minInertiaIdx + 1) * 2500 << std::endl
    << "Best nmi value: " << nmis[maxNmiIdx] << " at batchSize = " << (maxNmiIdx + 1) * 2500 << std::endl;
    return (minInertiaIdx + 1) * 2500;
}

void execute(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, const int k, const int batchSize,
std::vector<double>& times, std::vector<double>& speedups){
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "\n# threads = " << i << std::endl;
        auto* km = new k_means(images, labels, i, k, batchSize, MAX_ITER);
        std::cout << "fitting..." << std::endl;
        auto time = omp_get_wtime();
        auto [fst, snd] = km->fit(images, TOL);
        time = omp_get_wtime() - time;
        times[i-1] = time;
        delete km;
        km = nullptr;
        speedups[i-1] = times[0] / times[i-1];
        std::cout << "Time: " << time << " Speedup: " << speedups[i-1] << std::endl
        << "Inertia value: " << fst << std::endl
        << "NMI value: " << snd << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string filename = "../results/speedup.csv";
    std::ofstream file(filename);
    if(!file){
        std::cerr << "Error: Unable to open or create file: " << filename << std::endl;
        return 1;
    }
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
    int batchSize = 70000;
    MAX_N_THREADS = std::stoi(argv[1]);
    std::vector<double> times(MAX_N_THREADS);
    std::vector<double> speedups(MAX_N_THREADS);
    std::cout << "\nFinding best k..." << std::endl;
    k = findBestK(images, labels);
    std::cout << "\nFinding best batchSize..." << std::endl;
    batchSize = findBestBatchSize(images, labels, k);
    std::cout << "\nTesting..." << std::endl;
    std::cout << "\nFitting k_means..." << std::endl;
    execute(images, labels, k, batchSize, times, speedups);
    writeCSV(file, times, speedups);
    for(int i = 1; i <= MAX_N_THREADS; i++){
        std::cout << "# threads: " << i << std::endl
        << "Time: " << times[i-1] << " Speedup: " << speedups[i-1] << std::endl;
    }
    return 0;
}