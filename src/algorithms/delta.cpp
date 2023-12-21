#include "algorithms/delta.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

namespace algorithms {

    // vertex, value, delta, tent, buckets
    void relax(int vtx, float val, int d, float* t, int* b)
    {
        if (t[vtx] == -1 || val < t[vtx])
        {
            b[vtx] = val/d;
            t[vtx] = val;
        }
    }

    bool areAllBucketsEmpty(auto policy, int* buckets, int size) {
        return std::all_of(policy, buckets, buckets + size, [](int i) { return i == -1; });
    }

    std::vector<float> deltaStepping(sycl::queue& q, containers::CSR& csr, int source, int delta) {
// 
        auto policy = oneapi::dpl::execution::make_device_policy(q);
        auto size = csr.size;
        float* tent = sycl::malloc_shared<float>(size, q);
        int * buckets = sycl::malloc_shared<int>(size, q);
        int * tempBuckets = sycl::malloc_shared<int>(size, q);



        q.submit([&](sycl::handler& h) {
            h.parallel_for(size, [=](sycl::id<1> i) {
                tent[i] = -1;
                buckets[i] = -1;

                if(i == source) {
                    relax(i, 0, delta, tent, buckets);
                }
            });
        });

        while( !areAllBucketsEmpty(policy, buckets, size)) {

            std::copy(policy, buckets, buckets + size, tempBuckets);
            std::sort(policy, tempBuckets, tempBuckets + size);
            auto end = std::unique(policy, tempBuckets, tempBuckets + size);
            auto uniqueSize = std::distance(tempBuckets, end);

            q.submit([&](sycl::handler& h) {
                h.parallel_for(size, [=](sycl::id<1> i) {

                    for(int j = 0; j < uniqueSize; j++) {
                        if(buckets[i] == tempBuckets[j] && buckets[i] != -1) {
                            
                            // relax all light edges
                            
                        }
                    }
                });
            });
        }
        
        sycl::free(tent, q);
        sycl::free(buckets, q);

        return std::vector<float>{};
    };

    void test(sycl::queue&q) {

    }

} // namespace algorithms






//get light and heavy edges
//reduce to get numbers
//allocate data for light and heavy edges
//save indices of light and heavy edges per vertex
// write edges to light and heavy arrays
