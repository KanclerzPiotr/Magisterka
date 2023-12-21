#include "containers/CSR.hpp"

namespace containers {


    CSR::CSR(sycl::queue& q, int size) : q{q}, size{size} {
        row = sycl::malloc_shared<int>(size, q);
        col = sycl::malloc_shared<int>(size, q);
        val = sycl::malloc_shared<float>(size, q);
    }

    CSR::~CSR() {
        sycl::free(row, q);
        sycl::free(col, q);
        sycl::free(val, q);
    }

    void CSR::readFromFile(const std::string& filename) {
        
        if (filename.empty()) {
            return;
        }
    }

} // namespace containers

