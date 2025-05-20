import numpy as np
from PIL import Image
from numba import njit, prange

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

@njit(parallel=True)
def pad_image(src, dst, padding):
    h, w, _ = src.shape
    hp, wp, _ = dst.shape
    for i in prange(hp):
        for j in range(wp):
            if i >= padding and i < h + padding and j >= padding and j < w + padding:
                dst[i, j, 0] = src[i - padding, j - padding, 0]
                dst[i, j, 1] = src[i - padding, j - padding, 1]
                dst[i, j, 2] = src[i - padding, j - padding, 2]
                dst[i, j, 3] = src[i - padding, j - padding, 3]
            else:
                dst[i, j, 0] = 0
                dst[i, j, 1] = 0
                dst[i, j, 2] = 0
                dst[i, j, 3] = 0

@njit(parallel=True)
def convolve(padded, out, kernel, coeff, padding):
    h, w, _ = out.shape
    ks = kernel.shape[0]
    for i in prange(h):
        for j in range(w):
            r = g = b = a = 0
            # walk over the kernel window
            for ky in range(ks):
                for kx in range(ks):
                    pix = padded[i + ky, j + kx]
                    k = kernel[ky, kx]
                    r += pix[0] * k
                    g += pix[1] * k
                    b += pix[2] * k
                    a += pix[3] * k
            # normalize
            out[i, j, 0] = r // coeff
            out[i, j, 1] = g // coeff
            out[i, j, 2] = b // coeff
            out[i, j, 3] = a // coeff

def main():
    # load as RGBA
    img = Image.open("tester.bmp").convert("RGBA")
    src = np.array(img, dtype=np.uint8)
    h, w, _ = src.shape



    # allocate padded and output arrays
    padded = np.zeros((h + 2*PADDING, w + 2*PADDING, 4), dtype=np.uint8)
    out    = np.zeros_like(src)

    # first call compiles the functions; subsequent calls are fast
    pad_image(src, padded, PADDING)



    convolve(padded, out, KERNEL, COEFF, PADDING)

    for row in out:
        for x in row:
            print(x, end="")
        print("")

    # write result
    result = Image.fromarray(out, mode="RGBA")
    result.save("output.bmp")

if __name__ == "__main__":
    main()