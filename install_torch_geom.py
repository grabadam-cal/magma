#!/usr/bin/env python
"""
Assumes torch is already installed. 
"""
import torch
import os

if torch.version.cuda is not None:
	cuda_version_dot = torch.version.cuda

	cu_vers_nmbr = ''.join(cuda_version_dot.split('.'))

	cuda_version = 'cu'+cu_vers_nmbr

else:
	cuda_version = 'cpu'

torch_version = torch.__version__

#INSTALL SEQUENTIALLY
scatter_cmd = 'python -m pip install torch-scatter -f \
https://pytorch-geometric.com/whl/torch-%s+%s.html'%(torch_version, cuda_version)

sparse_cmd = 'python -m pip install torch-sparse -f \
https://pytorch-geometric.com/whl/torch-%s+%s.html'%(torch_version, cuda_version)

cluster_cmd = 'python -m pip install torch-cluster -f \
https://pytorch-geometric.com/whl/torch-%s+%s.html'%(torch_version, cuda_version)

spline_cmd = 'python -m pip install torch-spline-conv -f \
https://pytorch-geometric.com/whl/torch-%s+%s.html'%(torch_version, cuda_version)

geom_cmd = 'python -m pip install torch-geometric'

os.system(scatter_cmd)
os.system(sparse_cmd)
os.system(cluster_cmd)
os.system(spline_cmd)
os.system(geom_cmd)

print('Finished installing.')