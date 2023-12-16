```
conda create -n condaenv-torch python=3.11 ipykernel
conda activate condaenv-torch
```

Go to the following web-page:

    https://pytorch.org/get-started/locally/

Select the good configuration of your platform.

if Linux or Windows and you want GPU support (offered through NVIDIA)

    https://developer.nvidia.com/cuda-downloads

--> select Cuda version, but also install cuda support on your machine.

https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

using with GPU support, but be sure to install CUDA support **first**! from https://pytorch.org/get-started/locally/:

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

## Verification:

Type the following in, in a command shell or in a terminal of an IDE:

```python
import torch
x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()
```
