"""Small CUDA and NF4 checks; does not load or train a language model."""
import json
import os
import platform
from importlib.metadata import version
import torch
import bitsandbytes as bnb

if not torch.cuda.is_available():raise SystemExit('CUDA unavailable. Run this on the rented GPU, not the laptop check environment.')
device=0
major,minor=torch.cuda.get_device_capability(device)
if major>=12 and tuple(map(int,(torch.version.cuda or '0.0').split('.')[:2]))<(12,8):
    raise SystemExit('Blackwell requires an appropriate CUDA 12.8+ PyTorch build.')
dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
x=torch.ones((32,64),device='cuda:0',dtype=dtype)
y=x@x.T
torch.cuda.synchronize()
layer=bnb.nn.Linear4bit(64,64,bias=False,compute_dtype=dtype,quant_type='nf4',compress_statistics=True).to('cuda:0')
with torch.no_grad():z=layer(x)
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(z).all()
props=torch.cuda.get_device_properties(device)
report={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'bnb_cuda_override':os.environ.get('BNB_CUDA_VERSION'),'visible_gpu_count':torch.cuda.device_count(),'tested_device':props.name,'compute_capability':f'{major}.{minor}','vram_gib':round(props.total_memory/2**30,2),'dtype':str(dtype),'bitsandbytes':version('bitsandbytes'),'transformers':version('transformers'),'peft':version('peft'),'cuda_matmul':'passed','nf4_forward':'passed','language_model_training':'not run by this check'}
print(json.dumps(report,indent=2))
