# UdaciMed: Efficient Medical Diagnostics with Hardware-Aware AI

## Project Overview
This repository contains the final phase of the **UdaciMed** machine learning optimization pipeline. The goal of this project is to take an architecturally optimized PyTorch model for pneumonia detection and transform it into a production-ready, hardware-accelerated deployment format.

This project demonstrates how to bridge the gap between model training and real-world clinical deployment, focusing on strict Service Level Agreements (SLAs) for latency, throughput, memory footprint, and clinical safety.

## Key Features & Techniques
* **ONNX Conversion:** Exporting PyTorch models to the Open Neural Network Exchange (ONNX) format for cross-platform portability.
* **Hardware Acceleration:** 
  * **Mixed Precision (FP16):** Utilizing Tensor Cores on T4 GPUs to cut memory bandwidth and accelerate matrix multiplications.
  * **Dynamic Batching:** Optimizing batch sizes (1 to 64) to handle both real-time, single-patient inferences and high-volume offline batch processing.
* **Hardware-Specific Execution Providers:** Strategic analysis and deployment configurations for NVIDIA GPUs (CUDA/TensorRT), Intel CPUs (OpenVINO), and Edge devices (LiteRT/Core ML).
* **Clinical Validation:** Ensuring that hardware quantization (like FP16/INT8) do not degrade clinical sensitivity below the 98% SLA threshold (achieved 98.97%).

## Repository Structure
```text
├── 03_deployment_acceleration.ipynb  # Main Jupyter Notebook
├── utils/                            # Helper scripts (data loaders, model architectures, etc.)
├── results/                        # Exported ONNX models and performance metrics
├── README.md                 # Project overview and setup instructions
└── Report.md                    # Strategic deployment analysis and results