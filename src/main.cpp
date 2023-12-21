#include <iostream>

#include "algorithms/delta.hpp"
#include <sycl/sycl.hpp>


int main() {

    sycl::queue q{sycl::gpu_selector_v};
    std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    containers::CSR csr{q, 5};

    algorithms::deltaStepping(q, csr, 0, 5);


    return 0;
}
