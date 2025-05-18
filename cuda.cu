#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>


#pragma nv_diagnostic push
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma nv_diag_default 550
#pragma nv_diagnostic pop

constexpr int kernel_size = 5;
constexpr int padding     = kernel_size / 2;
constexpr int coeff       = 273;
constexpr int kernel[kernel_size * kernel_size] = 
      {1,  4,  7,  4, 1,
       4, 16, 26, 16, 4,
       7, 26, 41, 26, 7,
       4, 16, 26, 16, 4,
       1,  4,  7,  4, 1};

__constant__ int d_kernel[kernel_size * kernel_size];

// CUDA kernel: 5x5 Gaussian blur on uchar4 image with padding
__global__ void gaussianBlur(
    const uchar4* __restrict__ paddedData,
          uchar4* __restrict__ targetData,
    int width,
    int height,
    int pitchElems        // <-- elements per row
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int x     = ix + padding;
    int y     = iy + padding;
    int coord = y*pitchElems + x;    // now correct

    uint4 sum = make_uint4(0,0,0,0);
    for(int ky=-padding; ky<=padding; ++ky){
      for(int kx=-padding; kx<=padding; ++kx){
        int ki  = (ky+padding)*kernel_size + (kx+padding);
        int off = ky*pitchElems + kx;
        uchar4 pix = paddedData[coord + off];
        sum.x += d_kernel[ki]*pix.x;
        sum.y += d_kernel[ki]*pix.y;
        sum.z += d_kernel[ki]*pix.z;
        sum.w += d_kernel[ki]*pix.w;
      }
    }
    sum.x/=coeff; sum.y/=coeff; sum.z/=coeff; sum.w/=coeff;

    uchar4 out = {
      static_cast<unsigned char>(sum.x),
      static_cast<unsigned char>(sum.y),
      static_cast<unsigned char>(sum.z),
      static_cast<unsigned char>(sum.w)
    };
    targetData[coord] = out;
}


int main() {

    int width, height, channels;
    uchar4* host_data;

    host_data = reinterpret_cast<uchar4*>(
        stbi_load("tester.bmp", &width, &height, &channels, 4));

    const auto pitchWithPadding = width + 2 * padding;
    const auto sizeWithPadding = pitchWithPadding * (height + 2 * padding);
    // const auto sizeWithoutPadding = width * height;
    // const auto skipTopRows = pitchWithPadding * 2 + 2;
    // const auto bytesWithPadding = sizeWithPadding * sizeof(uchar4);
    const auto pitchWithPaddingInBytes = pitchWithPadding * sizeof(uchar4);
    const auto heightWithPadding = height + 2 * padding;
    const auto widthInBytes = width * sizeof(uchar4);

    // Allocate device memory
    uchar4* paddedData = nullptr;
    uchar4* targetData = nullptr;
    
    size_t devicePitchInBytes;
    size_t targetPitchInBytes;

    cudaMallocPitch(&paddedData, &devicePitchInBytes, pitchWithPaddingInBytes, heightWithPadding);
    cudaMallocPitch(&targetData, &targetPitchInBytes, pitchWithPaddingInBytes, heightWithPadding);

    const auto skipTopRows = 2 * (devicePitchInBytes / sizeof(uchar4)) + 2;
    
    cudaMemset2D(paddedData, devicePitchInBytes, 0, pitchWithPaddingInBytes, heightWithPadding);
    cudaMemcpy2D(paddedData + skipTopRows, devicePitchInBytes, host_data, widthInBytes, widthInBytes, height, cudaMemcpyHostToDevice);

    // cudaMemcpy(targetData, paddedData, devicePitchInBytes * heightWithPadding, cudaMemcpyDeviceToDevice);
 


    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(kernel), 0, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    gaussianBlur<<<grid, block>>>(
        paddedData,
        targetData,
        width,
        height,
        devicePitchInBytes / sizeof(uchar4));
    
    cudaMemcpy2D(host_data, widthInBytes, targetData + skipTopRows, targetPitchInBytes, widthInBytes, height, cudaMemcpyDeviceToHost);

    stbi_write_bmp("output.bmp", width, height, 4, host_data);

    return 0;
}