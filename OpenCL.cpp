// gaussian_blur_opencl.cpp

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int kernel_size = 5;
constexpr int padding = kernel_size / 2;
constexpr int coeff = 273;
const int h_kernel[kernel_size * kernel_size] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};

// OpenCL kernel source
const char* kernelSource = R"CLC(
__kernel void gaussianBlur(
    __global const uchar4* paddedData,
    __global uchar4* targetData,
    __constant int* d_kernel,
    const int width,
    const int height,
    const int pitchElems,
    const int padding,
    const int kernel_size,
    const int coeff)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    int x = ix + padding;
    int y = iy + padding;
    int coord = y * pitchElems + x;

    uint4 sum = (uint4)(0, 0, 0, 0);
    for (int ky = -padding; ky <= padding; ky++) {
        for (int kx = -padding; kx <= padding; kx++) {
            int ki  = (ky + padding) * kernel_size + (kx + padding);
            int off = ky * pitchElems + kx;
            uchar4 pix = paddedData[coord + off];
            sum.x += d_kernel[ki] * pix.x;
            sum.y += d_kernel[ki] * pix.y;
            sum.z += d_kernel[ki] * pix.z;
            sum.w += d_kernel[ki] * pix.w;
        }
    }
    sum.x /= coeff; sum.y /= coeff; sum.z /= coeff; sum.w /= coeff;
    targetData[coord] = (uchar4)(sum.x, sum.y, sum.z, sum.w);
    // targetData[coord] = paddedData[coord];
}
)CLC";

cl::Device getCudaDevice() {

    cl::Platform platform;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for(auto& plat : platforms)
    {
        if (plat.getInfo<CL_PLATFORM_NAME>().find("NVIDIA") != std::string::npos)
        {
            std::vector<cl::Device> devices;
            plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            return devices.front();
        }

    }
    return cl::Device();
}

inline std::ostream& operator<<(std::ostream& os, const cl_uchar4 &v) {
    return os << '('
              << std::setw(3) << static_cast<int>(v.s[0]) << ", "
              << std::setw(3) << static_cast<int>(v.s[1]) << ", "
              << std::setw(3) << static_cast<int>(v.s[2]) << ", "
              << std::setw(3) << static_cast<int>(v.s[3])
              << ')';
}

int main() {
    int width, height, channels;
    cl_uchar4* host_data = reinterpret_cast<cl_uchar4*>(
        stbi_load("obraz.bmp", &width, &height, &channels, 4));


    auto widthInBytes = width * sizeof(cl_uchar4);
    auto paddedWidth = width + 2 * padding;
    auto paddedWidthInBytes = paddedWidth * sizeof(cl_uchar4);
    auto paddedHeight = height + 2 * padding;
    auto totalElems = paddedWidth * paddedHeight;
    auto totalSize = totalElems * sizeof(cl_uchar4);

    cl::Device device = getCudaDevice();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer paddedData(context, CL_MEM_READ_ONLY, totalSize);
    cl::Buffer targetData(context, CL_MEM_WRITE_ONLY, totalSize);
    cl::Buffer d_kernel_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_kernel), (void*)h_kernel);

    cl_uchar4 zero{0, 0, 0, 0};
    queue.enqueueFillBuffer(paddedData, zero, 0, totalSize, nullptr);

    cl::array<size_t, 3> skipTopRows{padding * sizeof(cl_uchar4), padding, 0};
    cl::array<size_t, 3> hostOffset{0, 0, 0};
    cl::array<size_t, 3> region{width * sizeof(cl_uchar4), height, 1};

    queue.enqueueWriteBufferRect(paddedData, CL_TRUE, skipTopRows, hostOffset, region, paddedWidthInBytes, 0, widthInBytes, 0, host_data);

    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, "gaussianBlur");
    kernel.setArg(0, paddedData);
    kernel.setArg(1, targetData);
    kernel.setArg(2, d_kernel_buf);
    kernel.setArg(3, width);
    kernel.setArg(4, height);
    kernel.setArg(5, paddedWidth);
    kernel.setArg(6, padding);
    kernel.setArg(7, kernel_size);
    kernel.setArg(8, coeff);

    cl::NDRange global(width, height);
    cl::NDRange local(16, 16);

    queue.enqueueNDRangeKernel(kernel, {}, global, cl::NullRange);
    queue.finish();

    queue.enqueueReadBufferRect(targetData, CL_TRUE, skipTopRows, hostOffset, region, paddedWidthInBytes, 0, widthInBytes, 0, host_data);
    stbi_write_bmp("output.bmp", width, height, 4, host_data);

    return 0;
}
