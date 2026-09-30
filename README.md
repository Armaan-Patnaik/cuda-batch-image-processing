# CUDA Batch Image Processing

## Project summary

This project performs GPU-accelerated batch image processing using a custom CUDA kernel.

The program applies a configurable box blur to a large batch of grayscale PGM images. A single CUDA kernel launch processes all pixels from all images, with one GPU thread responsible for one output pixel.

The included test pipeline generates 200 small 128x128 images and processes all 200 in one execution.

## GPU computation

`BoxBlurBatchKernel` performs the image processing on the CUDA GPU.

For every pixel, the kernel reads a square neighborhood around that pixel, computes the average intensity, and writes the blurred value to the output image.

The input images are copied to GPU memory with `cudaMemcpy`, processed by the CUDA kernel, and copied back with `cudaMemcpy`.

CUDA events are used to measure GPU kernel execution time and throughput.

## Command line arguments

```text
-i <directory>   Input directory containing PGM images
-o <directory>   Output directory
-n <count>       Number of images to process
-r <radius>      Blur radius
-t <threads>     Threads per CUDA block
```

Example:

```bash
./batch_blur.exe -i dataset -o output -n 200 -r 2 -t 256
```

## Build

```bash
make clean
make build
```

## Generate data

```bash
python3 generate_dataset.py \
    --out dataset \
    --count 200 \
    --width 128 \
    --height 128
```

## Run

```bash
./batch_blur.exe \
    -i dataset \
    -o output \
    -n 200 \
    -r 2 \
    -t 256
```

## One-command execution

```bash
./run.sh
```

The script:

1. Generates 200 test images.
2. Builds the CUDA program.
3. Processes all 200 images on the GPU.
4. Stores the console log in `artifacts/execution.log`.
5. Saves before/after copies of the first image in `artifacts/`.

## Results

The execution log reports:

- Number of images processed
- Image dimensions
- CUDA threads per block
- CUDA blocks in the grid
- GPU kernel execution time
- GPU throughput in megapixels per second
- Successful completion

The project uses a custom CUDA kernel rather than a CPU-only implementation.
