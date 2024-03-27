#include "algorithms/delta.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <iostream>
namespace algorithms {

    constexpr int PUT_IN_R = std::numeric_limits<int>::max() - 2;
    constexpr int FULLY_RELAXED = std::numeric_limits<int>::max() - 1;

    // vertex, value, delta, tent, buckets
    void relax(int vtx, float val, int d, float* t, int* b)
    {
        if (val < t[vtx])
        {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> ref_t(t[vtx]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref_b(b[vtx]);
            ref_b.fetch_min((int)val/d);
            ref_t.fetch_min(val);
        }
    }

    bool areAllBucketsEmpty(auto policy, int* buckets, int size) {
        return std::all_of(policy, buckets, buckets + size, [](int i) { return i == FULLY_RELAXED; });
    }

    void relaxLightEdges(sycl::id<1> i, int delta, float* tent, int* buckets, int* row, int* col, float* val) {
        for(int j = row[i]; j < row[i+1]; j++) {
            if (val[j] < delta) { // light edge
                relax(col[j], tent[i] + val[j], delta, tent, buckets);
            }
        }
    }
    
    void relaxHeavyEdges(sycl::id<1> i, int delta, float* tent, int* buckets, int* row, int* col, float* val) {
        for(int j = row[i]; j < row[i+1]; j++) {
            if (val[j] >= delta) { // heavy edge
                relax(col[j], tent[i] + val[j], delta, tent, buckets);
            }
        }
    }

    std::vector<float> deltaStepping(sycl::queue& q, containers::CSR& csr, int source, int delta) {
// 
        auto policy = oneapi::dpl::execution::make_device_policy(q);
        auto size = csr.vertices;
        auto row = csr.row;
        auto col = csr.col;
        auto val = csr.val;
        float* tent = sycl::malloc_shared<float>(size, q);
        int * buckets = sycl::malloc_shared<int>(size, q);
        int x;

        q.submit([&](sycl::handler& h) {
            h.parallel_for(size, [=](sycl::id<1> i) {
                tent[i] = std::numeric_limits<float>::max();
                buckets[i] = std::numeric_limits<int>::max();

                if(i == source) {
                    relax(i, 0, delta, tent, buckets);
                }
            });
        }).wait();

        while( !areAllBucketsEmpty(policy, buckets, size)) {
            
            auto bucket = std::min_element(policy, buckets, buckets + size);
            int bucketValue;
            int newBucketValue;
            q.copy<int>(bucket, &bucketValue, 1).wait();
            // while i'th bucket is not empty
            do {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(size, [=](sycl::id<1> i) {
                        if(buckets[i] == bucketValue) {
                            
                            relaxLightEdges(i, delta, tent, buckets, row, col, val);
                            buckets[i] = PUT_IN_R;

                        }
                    });
                }).wait();
                
                bucket = std::min_element(policy, buckets, buckets + size);
                q.copy<int>(bucket, &newBucketValue, 1).wait();
            } while (bucketValue == newBucketValue);

            bucketValue = newBucketValue;

            q.submit([&](sycl::handler& h) {
                h.parallel_for(size, [=](sycl::id<1> i) {
                    if(buckets[i] == PUT_IN_R) {
                        relaxHeavyEdges(i, delta, tent, buckets, row, col, val);
                        buckets[i] = FULLY_RELAXED;

                    }
                });
            }).wait();

        }   
        
        std::vector<float> result(size);
        q.memcpy(result.data(), tent, size * sizeof(float)).wait();
        sycl::free(tent, q);
        sycl::free(buckets, q);

        return result;
    }

} // namespace algorithms

