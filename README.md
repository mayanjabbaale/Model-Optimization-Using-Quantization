# Model-Optimization-Using-Quantization

This code explores the different types of quantization, and apply both post training quantization (PTQ) and quantization aware training (QAT) on a simple example using CIFAR-10 and ResNet18. The process achieves a 75% reduction in space and 16% reduction in GPU latency with only 1% drop in accuracy.

##After basic training.
```bash
Size (full): 44.77 MB
Accuracy (full): 80.53%
Latency (full, on CPU): 804.16 ± 57.55 ms
Latency (full, on GPU): 16.39 ± 0.30 ms
```

## Post Training Quantization (PTQ)
The basic flow is as follow:

1. Export the model to to a stable, backend-agnostic format that’s suitable for transformations, optimizations, and deployment.
2. Define the quantizer that will prepare the model for quantization. Here I used the X86 for CPU deployments, but there is a simple variant that works better for mobile and edge devices working on ARM CPUs.
3. Preparing the model for quantization. For example, folding batch-norm into preceding conv2d operators, and inserting observers in appropriate places to collect activation statistics needed for calibration.
4. Running inference on calibration data to collect activation statistics
5. Converts calibrated model to a quantized model. While the quantized model already takes less space, it is not yet optimized for the final deployment.

```python
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

Results:
Size (optimized): 11.26 MB
Accuracy (optimized): 79.53%
Latency (optimized, on CPU): 782.53 ± 51.36 ms
Latency (optimized, on GPU): 13.80 ± 0.28 ms
```
## Quantization Aware Training (QAT)
In QAT the basic flow is very similiar to PTQ, the main difference is the replacement of the calibration step that collects activation statistics with a much longer fine-tuning step which fine-tunes the model considering the quantization constraints. The collection of activation statistics also happens, as part of the fine-tuning process.

```python
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)

import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

Results:
Size (optimized): 11.26 MB
Accuracy (optimized): 79.54%
Latency (optimized, on CPU): 831.76 ± 39.63 ms
Latency (optimized, on GPU): 13.71 ± 0.24 ms
```


