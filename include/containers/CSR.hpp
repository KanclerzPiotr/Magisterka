#pragma once

#include <sycl/sycl.hpp>
#include <string>

namespace containers {

struct CSR
{
    int* row;
    int* col;
    float* val;
    int size;
    sycl::queue& q;
    
    CSR(sycl::queue& q, int size);
    ~CSR();
    void readFromFile(const std::string& filename);

    

};

} // namespace containers