#pragma once

#include <sycl/sycl.hpp>
#include <string_view>

namespace containers {

struct AdjeacenacyMatrix
{
    float* data;
    int vertices;
    int edges;
    sycl::queue& q;
    
    explicit AdjeacenacyMatrix(sycl::queue& q);
    ~AdjeacenacyMatrix();
    void readFromFile(std::string_view filename);
    void print();

private:

    void allocateData();

    

};

} // namespace containers