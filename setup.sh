pip uninstall -y torch
pip uninstall -y torch-scatter
pip uninstall -y torch-sparse
pip uninstall -y torch-cluster
pip uninstall -y torch-spline-conv
pip uninstall -y torch-geometric
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric
pip install tensorboardX
pip install geopy
pip install pandarallel
