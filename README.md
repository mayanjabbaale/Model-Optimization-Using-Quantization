# Model-Optimization-Using-Quantization

This code explores the different types of quantization, and apply both post training quantization (PTQ) and quantization aware training (QAT) on a simple example using CIFAR-10 and ResNet18. The process achieves a 75% reduction in space and 16% reduction in GPU latency with only 1% drop in accuracy.

## Post Training Quantization (PTQ)
The basic flow is as follow:

1. Export the model to to a stable, backend-agnostic format that’s suitable for transformations, optimizations, and deployment.
2. Define the quantizer that will prepare the model for quantization. Here I used the X86 for CPU deployments, but there is a simple variant that works better for mobile and edge devices working on ARM CPUs.
3. Preparing the model for quantization. For example, folding batch-norm into preceding conv2d operators, and inserting observers in appropriate places to collect activation statistics needed for calibration.
4. Running inference on calibration data to collect activation statistics
5. Converts calibrated model to a quantized model. While the quantized model already takes less space, it is not yet optimized for the final deployment.

'''
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
'''
