# CUDA Accelerated AI & Computer Vision Tasks

A collection of CUDA implementations designed to eliminate CPU bottlenecks in deep learning inference pipelines and computer vision workloads by offloading memory operations and computation entirely to the GPU.

---

## Overview

In standard machine learning and computer vision pipelines, heavy compute runs on the GPU while post-processing operations (such as tensor reductions, masking, and filtering) fall back to the host CPU via NumPy or OpenCV. This creates costly host-to-device memory roundtrips and CPU processing bottlenecks.

This repository demonstrates GPU-native workflows across two implementations:
1. **GPU-Accelerated Semantic Segmentation Post-Processing**: Fuses per-pixel channel argmax reduction and binary RGB masking into a single inline CUDA C++ kernel using PyTorch.
2. **2D Gaussian Blur Kernel (CUDA vs. CPU Benchmark)**: Compares a multi-threaded 2D grid CUDA image filter against an OpenCV C++ CPU implementation, achieving a **~117x speedup** with bit-exact accuracy.

---

## Tasks

### 1. GPU-Accelerated Semantic Segmentation Post-Processing

* **Model**: Pretrained DeepLabV3-ResNet50 (Pascal VOC, 21 classes).
* **Inference Runtime**: Exported to ONNX and executed via ONNX Runtime with `CUDAExecutionProvider` to yield raw logits on GPU memory `(1, 21, 520, 520)`.
* **Custom CUDA Kernel (`seg_ext_min`)**:
  * Compiled inline at runtime via `torch.utils.cpp_extension.load_inline`.
  * Runs a 1D grid with 256 threads per block over all $H \times W$ pixels.
  * Iterates across 21 class channels per pixel to calculate argmax in place.
  * Emits an 8-bit unsigned integer (`uint8`) class mask.
  * Isolates a specified target class (e.g., Person = `15`) and masks non-target pixels to black directly on the RGB image.
* **Validation**: Output masks and composited RGB images verify against a NumPy CPU reference with 100% exact parity (`np.array_equal == True`).

```
Input RGB (520x520) ──► ONNX Runtime GPU ──► Logits [1, 21, 520, 520]
                                                     │
                                                     ▼
                                         Custom CUDA Kernel
                                         ├── Argmax across 21 classes
                                         ├── Write uint8 class mask
                                         └── Mask non-target RGB pixels
                                                     │
                                                     ▼
                                        `masked.png` & `mask_ids.png`
```

---

### 2. 2D Gaussian Blur Filter (CUDA vs. CPU Benchmark)

* **Kernel**: $3 \times 3$ normalized Gaussian convolution filter:
  $$\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
* **CPU Implementation (`cpu_blur.cpp`)**: Sequential nested loop over interior image coordinates using standard OpenCV Mat accessors.
* **GPU Implementation (`gpu_blur.cu`)**:
  * 2D execution grid: `dim3 block(16, 16)` and `dim3 grid((w + 15) / 16, (h + 15) / 16)`.
  * Threads at image borders copy input pixels directly; interior threads perform the weighted 9-pixel accumulation and division.
  * Timed via `cudaEvent_t` recording for hardware-accurate execution measurement.

#### Benchmark Results (NVIDIA T4 GPU)

| Method | Implementation | Execution Time | Speedup | Accuracy Match |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | C++ / OpenCV | `12.000 ms` | 1.0x (Baseline) | Reference |
| **GPU** | Native CUDA C++ | **`0.102 ms`** | **~117x faster** | **100.00%** (50,325 / 50,325 pixels) |

---

## Requirements & Environment

Both notebooks run natively in Google Colab with a GPU runtime (T4 or higher).

### System Dependencies
- NVIDIA GPU + CUDA Toolkit $\ge$ 12.0
- GCC / G++ with C++17 support
- OpenCV 4 development packages:
  ```bash
  sudo apt update && sudo apt install -y libopencv-dev pkg-config
  ```

### Python Dependencies
```bash
pip install torch torchvision onnx onnxscript onnxruntime-gpu ninja opencv-python numpy pillow nvcc4jupyter
```

---

## Running the Tasks

### Task 1: Segmentation Pipeline
1. Place a test image named `img.jpeg` in the working directory.
2. Run the segmentation notebook sequentially to export the model, compile the inline kernel, and generate `masked.png` and `mask_ids.png`.

### Task 2: Gaussian Blur Benchmark
1. Place a test image named `download.jpg` in the working directory.
2. Run the Gaussian blur notebook to compile `cpu_blur.cpp` and `gpu_blur.cu`, execute both programs, and run the pixel-difference verification script.
