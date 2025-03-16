# Non-rigid Deformation Experiment
## Environment

```
sudo apt install libsuitesparse-dev
```

```
conda create --name nicp -y python=3.10
conda activate nicp

pip install jax
python -m pip install libigl fcpw trimesh scikit-sparse Pillow tqdm icecream polyscope scipy==1.15.0 numpy==1.26.4 sparseqr==1.2.1
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
