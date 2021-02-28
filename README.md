# intrinsic motivation experiments

# 1 atari pacman



# dependences
cmake python3 python3-pip

**basic python libs**
pip3 install numpy matplotlib torch torchviz pillow opencv-python networkx

**graph neural networks**
when CPU only :

pip3 install torch_geometric torch_sparse torch_scatter


for CUDA different packages are reuired :
- this is for cuda 10.2
- and pytorch 1.6

detect pytorch cuda version

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"

pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-geometric

example for torch 1.7.0 and cuda 10.2

pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip3 install torch-geometric


see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

**environments**
pip3 install  gym pybullet pybulletgym 'gym[atari]' 'gym[box2d]' gym-super-mario-bros gym_2048