# CUDA Batch Image Processing

## Project summary

This project performs GPU-accelerated batch image processing using a custom CUDA kernel.

The program applies a configurable box blur to a batch of grayscale PGM images. A CUDA kernel processes all pixels from all images in parallel, with one GPU thread responsible for one output pixel.

The test pipeline generates 200 grayscale images of size 128x128 and processes the complete batch in a single execution.

## GPU computation

`BoxBlurBatchKernel` performs the image processing on the CUDA GPU.

For every output pixel, one GPU thread:

1. Identifies the corresponding image and pixel.
2. Reads a square neighborhood around the pixel.
3. Handles image boundaries by clamping coordinates.
4. Computes the average intensity of the neighborhood.
5. Writes the blurred pixel to the output image.

The input images are stored in contiguous host memory, copied to GPU memory using `cudaMemcpy`, processed by the CUDA kernel, and copied back to host memory.

CUDA events are used to measure the GPU kernel execution time.

## Command line arguments

```text
-i <directory>   Input directory containing PGM images
-o <directory>   Output directory
-n <count>       Number of images to process
-r <radius>      Blur radius
-t <threads>     Threads per CUDA block
