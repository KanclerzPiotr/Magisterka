#include "containers/CSR.hpp"
#include <fstream>

namespace containers {


    CSR::CSR(sycl::queue& q) : q{q} {}

    CSR::~CSR() {
        sycl::free(row, q);
        sycl::free(col, q);
        sycl::free(val, q);
    }

    void CSR::readFromFile(std::string_view filename) {
        
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
            row[u+1]++;
            col[i] = v;
            val[i] = value;
        }
    }

    void CSR::fromAdjeacenacyMatrix(const AdjeacenacyMatrix& matrix)
    {
        vertices = matrix.vertices;
        edges = matrix.edges;
        allocateData();

        int edge = 0;
        for(int i = 0; i < vertices; i++) {
            for(int j = 0; j < vertices; j++) {
                if(matrix.data[i * vertices + j] != 0) {
                    col[edge] = j;
                    val[edge] = matrix.data[i * vertices + j];
                    edge++;
                    row[i+1]= edge;
                }
            }
        }
    }

    void CSR::print()
    {
        for(int i =0; i < vertices; i++)
        {
            std::cout << "row[" << i << "] = " << row[i] << std::endl;
            for(int j = row[i]; j < row[i+1]; j++)
            {
                std::cout << "\tcol[" << col[j] << "] = " << val[j] << std::endl;
            }
        }
    }

    void CSR::allocateData() {
        row = sycl::malloc_shared<int>(vertices + 1, q);
        col = sycl::malloc_shared<int>(edges, q);
        val = sycl::malloc_shared<float>(edges, q);
    }

} // namespace containers

