#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error during " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool ReadPgm(const fs::path& path,
             int* width,
             int* height,
             std::vector<unsigned char>* pixels) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }

    std::string magic;
    int max_value = 0;
    input >> magic >> *width >> *height >> max_value;

    if (!input || magic != "P5" || max_value != 255 ||
        *width <= 0 || *height <= 0) {
        return false;
    }

    input.get();

    pixels->resize(static_cast<size_t>(*width) *
                   static_cast<size_t>(*height));
    input.read(reinterpret_cast<char*>(pixels->data()),
               static_cast<std::streamsize>(pixels->size()));

    return input.gcount() ==
           static_cast<std::streamsize>(pixels->size());
}

bool WritePgm(const fs::path& path,
              int width,
              int height,
              const std::vector<unsigned char>& pixels) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }

    output << "P5\n"
           << width << ' ' << height << "\n255\n";
    output.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));

    return static_cast<bool>(output);
}

std::vector<fs::path> FindPgmFiles(const fs::path& directory) {
    std::vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".pgm") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

int ParseInt(int argc, char** argv, const std::string& option, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == option) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

std::string ParseString(int argc,
                        char** argv,
                        const std::string& option,
                        const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == option) {
            return argv[i + 1];
        }
    }
    return fallback;
}

}  // namespace

__global__ void BoxBlurBatchKernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width,
                                   int height,
                                   int image_count,
                                   int radius) {
    const int global_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int pixels_per_image = width * height;
    const int total_pixels = pixels_per_image * image_count;

    if (global_index >= total_pixels) {
        return;
    }

    const int image_index = global_index / pixels_per_image;
    const int pixel_index = global_index % pixels_per_image;

    const int x = pixel_index % width;
    const int y = pixel_index / width;

    int sum = 0;
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int sample_x =
                (x + dx < 0) ? 0 :
                (x + dx >= width) ? width - 1 :
                x + dx;

            const int sample_y =
                (y + dy < 0) ? 0 :
                (y + dy >= height) ? height - 1 :
                y + dy;

            const int sample_index =
                image_index * pixels_per_image +
                sample_y * width +
                sample_x;

            sum += input[sample_index];
            ++count;
        }
    }

    output[global_index] =
        static_cast<unsigned char>((sum + count / 2) / count);
}

int main(int argc, char** argv) {
    const std::string input_directory =
        ParseString(argc, argv, "-i", "dataset");
    const std::string output_directory =
        ParseString(argc, argv, "-o", "output");
    const int requested_count =
        ParseInt(argc, argv, "-n", 200);
    const int radius =
        ParseInt(argc, argv, "-r", 2);
    const int threads =
        ParseInt(argc, argv, "-t", 256);

    if (requested_count <= 0 || radius < 0 || threads <= 0) {
        std::cerr << "Invalid command line arguments.\n";
        return EXIT_FAILURE;
    }

    if (!fs::exists(input_directory)) {
        std::cerr << "Input directory does not exist: "
                  << input_directory << '\n';
        return EXIT_FAILURE;
    }

    const auto all_files = FindPgmFiles(input_directory);

    if (all_files.empty()) {
        std::cerr << "No .pgm input files were found.\n";
        return EXIT_FAILURE;
    }

    const int image_count =
        std::min(requested_count,
                 static_cast<int>(all_files.size()));

    std::vector<unsigned char> host_input;
    std::vector<unsigned char> host_output;

    int width = 0;
    int height = 0;

    std::vector<unsigned char> first_image;

    for (int i = 0; i < image_count; ++i) {
        int image_width = 0;
        int image_height = 0;
        std::vector<unsigned char> image;

        if (!ReadPgm(all_files[i],
                     &image_width,
                     &image_height,
                     &image)) {
            std::cerr << "Failed to read PGM: "
                      << all_files[i] << '\n';
            return EXIT_FAILURE;
        }

        if (i == 0) {
            width = image_width;
            height = image_height;
        } else if (image_width != width ||
                   image_height != height) {
            std::cerr << "All images must have the same dimensions.\n";
            return EXIT_FAILURE;
        }

        host_input.insert(
            host_input.end(),
            image.begin(),
            image.end());
    }

    host_output.resize(host_input.size());

    fs::create_directories(output_directory);

    const size_t total_bytes = host_input.size();

    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;

    CheckCuda(cudaMalloc(
                  reinterpret_cast<void**>(&device_input),
                  total_bytes),
              "cudaMalloc(device_input)");

    CheckCuda(cudaMalloc(
                  reinterpret_cast<void**>(&device_output),
                  total_bytes),
              "cudaMalloc(device_output)");

    CheckCuda(cudaMemcpy(
                  device_input,
                  host_input.data(),
                  total_bytes,
                  cudaMemcpyHostToDevice),
              "copying input images to GPU");

    const int total_pixels =
        width * height * image_count;

    const int block_count =
        (total_pixels + threads - 1) / threads;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CheckCuda(cudaEventCreate(&start_event),
              "creating start event");
    CheckCuda(cudaEventCreate(&stop_event),
              "creating stop event");

    std::cout << "CUDA batch image processing\n";
    std::cout << "Input directory: " << input_directory << '\n';
    std::cout << "Output directory: " << output_directory << '\n';
    std::cout << "Images processed: " << image_count << '\n';
    std::cout << "Image size: " << width << "x" << height << '\n';
    std::cout << "Blur radius: " << radius << '\n';
    std::cout << "Threads per block: " << threads << '\n';
    std::cout << "Blocks in grid: " << block_count << '\n';

    CheckCuda(cudaEventRecord(start_event),
              "recording start event");

    BoxBlurBatchKernel<<<block_count, threads>>>(
        device_input,
        device_output,
        width,
        height,
        image_count,
        radius);

    CheckCuda(cudaGetLastError(),
              "launching BoxBlurBatchKernel");

    CheckCuda(cudaEventRecord(stop_event),
              "recording stop event");
    CheckCuda(cudaEventSynchronize(stop_event),
              "waiting for GPU kernel");

    float kernel_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(
                  &kernel_ms,
                  start_event,
                  stop_event),
              "measuring GPU kernel");

    CheckCuda(cudaMemcpy(
                  host_output.data(),
                  device_output,
                  total_bytes,
                  cudaMemcpyDeviceToHost),
              "copying results from GPU");

    std::cout << "GPU kernel execution time: "
              << kernel_ms << " ms\n";

    const double megapixels =
        static_cast<double>(total_pixels) / 1.0e6;
    const double seconds =
        static_cast<double>(kernel_ms) / 1000.0;

    if (seconds > 0.0) {
        std::cout << "GPU throughput: "
                  << megapixels / seconds
                  << " megapixels/s\n";
    }

    for (int i = 0; i < image_count; ++i) {
        const fs::path output_path =
            fs::path(output_directory) /
            all_files[i].filename();

        const size_t offset =
            static_cast<size_t>(i) *
            static_cast<size_t>(width) *
            static_cast<size_t>(height);

        std::vector<unsigned char> image(
            host_output.begin() + offset,
            host_output.begin() + offset +
                static_cast<size_t>(width) *
                static_cast<size_t>(height));

        if (!WritePgm(
                output_path,
                width,
                height,
                image)) {
            std::cerr << "Failed to write: "
                      << output_path << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "Processing completed successfully.\n";

    CheckCuda(cudaEventDestroy(start_event),
              "destroying start event");
    CheckCuda(cudaEventDestroy(stop_event),
              "destroying stop event");
    CheckCuda(cudaFree(device_input),
              "freeing device input");
    CheckCuda(cudaFree(device_output),
              "freeing device output");

    return EXIT_SUCCESS;
}
