#include "containers/AdjeacenacyMatrix.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace containers {


    AdjeacenacyMatrix::AdjeacenacyMatrix(sycl::queue& q) : q{q} {}

    AdjeacenacyMatrix::~AdjeacenacyMatrix() {
        sycl::free(data, q);
    }

    void AdjeacenacyMatrix::readFromFile(std::string_view filename) {
        
        std::ifstream file(filename.data());
        if(!file.is_open()) {
            throw std::runtime_error("File not found");
        }

        file >> vertices >> edges;
        allocateData();

        for(int i = 0; i < edges; i++) {
            int u;
            int v;
            float value;
            file >> u >> v >> value;
            data[u * vertices + v] = value;
            data[v * vertices + u] = value;
        }
        edges *= 2;
    }

    void AdjeacenacyMatrix::print()
    {
        for(int i =0; i < vertices; i++)
        {
            for(int j = 0; j < vertices; j++)
            {
                std::cout << std::setw(4);
                std::cout << data[i * vertices + j];
            }
            std::cout << std::endl;
        }
    
    }

    void AdjeacenacyMatrix::allocateData() {
        data = sycl::malloc_shared<float>(vertices * vertices, q);
    }

} // namespace containers

