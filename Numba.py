from PIL import Image
import numpy as np
from numba import cuda
import time

# parameters
KERNEL_SIZE = 5
PADDING = KERNEL_SIZE // 2
COEFF = 273
KERNEL = np.array([
    [ 1,  4,  7,  4, 1],
    [ 4, 16, 26, 16, 4],
    [ 7, 26, 41, 26, 7],
    [ 4, 16, 26, 16, 4],
    [ 1,  4,  7,  4, 1]
], dtype=np.int32)

@cuda.jit
def copy_pad_image(src, dst, padding):
    i, j = cuda.grid(2)
    h, w, _ = dst.shape
    if i >= padding and i < h - padding and j >= padding and j < w - padding:
        for c in range(4):
            dst[i, j, c] = src[i - padding, j - padding, c]


@cuda.jit
def convolve(padded, out, kernel, coeff):
    i, j = cuda.grid(2)
    h, w, _ = out.shape
    ks = kernel.shape[0]
    r = g = b = a = 0
    if i < h and j < w:
        for ky in range(ks):
            for kx in range(ks):
                pix = padded[i + ky, j + kx]
                k = kernel[ky, kx]
                r += pix[0] * k
                g += pix[1] * k
                b += pix[2] * k
                a += pix[3] * k
                out[i, j, 0] = r // coeff
                out[i, j, 1] = g // coeff
                out[i, j, 2] = b // coeff
                out[i, j, 3] = a // coeff

def main():
    # load as RGBA
    img = Image.open("obraz.bmp").convert("RGBA")
    src = np.array(img, dtype=np.uint8)
    h, w, _ = src.shape

    padded = np.zeros((h + 2* PADDING, w + 2* PADDING, 4), dtype=np.uint8)

    d_padded = cuda.to_device(padded)
    d_out = cuda.to_device(src)
    d_kernel = cuda.to_device(KERNEL)

    threadsperblock = (16, 16)
    blockspergrid_x = (d_padded.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (d_padded.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    copy_pad_image[blockspergrid, threadsperblock](d_out, d_padded, PADDING)

    convolve[blockspergrid, threadsperblock](d_padded, d_out, d_kernel, COEFF)

    out = d_out.copy_to_host()

    result = Image.fromarray(out, mode="RGBA")
    result.save("output.bmp")

if __name__ == "__main__":
    main()