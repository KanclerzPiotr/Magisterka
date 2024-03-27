#pragma once

#include <sycl/sycl.hpp>
#include <string_view>
#include "containers/AdjeacenacyMatrix.hpp"


namespace containers {

struct CSR
{
    int* row;
    int* col;
    float* val;
    int vertices;
    int edges;
    sycl::queue& q;
    
    explicit CSR(sycl::queue& q);
    ~CSR();
    void readFromFile(std::string_view filename);
    void fromAdjeacenacyMatrix(const AdjeacenacyMatrix& matrix);
    void print();

private:

    void allocateData();

    

};

} // namespace containers