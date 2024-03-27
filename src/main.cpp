#include <iostream>

#include "algorithms/delta.hpp"
#include <sycl/sycl.hpp>

int main() {

    sycl::queue q{sycl::default_selector_v};
    std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // containers::CSR csr{q, 5};
    containers::AdjeacenacyMatrix matrix{q};
    containers::CSR csr{q};
    matrix.readFromFile("../../data/graph.txt");
    // matrix.print();
    csr.fromAdjeacenacyMatrix(matrix);
    // csr.print();

    auto vector = algorithms::deltaStepping(q, csr, 0, 25);

    for(auto v : vector) {
        std::cout << v << " ";
    }

    return 0;
}
