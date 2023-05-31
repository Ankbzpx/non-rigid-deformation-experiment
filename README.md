# Non-rigid Deformation Experiment
## Environment
```
conda create --name nicp -y python=3.10
conda activate nicp

# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d

yes | pip install polyscope icecream trimesh pillow scipy
yes | python -m pip install libigl
yes | pip install "jax[cpu]"
```

## Steps
Save polygon group to template mesh
```
python save_polygon_group.py
```

Match using ARAP + NICP
```
python non_rigid_deformation_arap.py
```

## Note
- Models are proprietary hence not included
- Unrelated files are test code
