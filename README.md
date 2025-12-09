# Prism Camera Calibration

This repository contains a PyTorch-based differentiable ray tracing library for calibrating cameras viewing through planar mirrors, refracting surfaces and other geometries that can be built using them, like prism mirrors and fish tranks.
The repository is under development.

## Notebooks

### 1. `runme_pytorch_raytracing_notebook.ipynb`

A tutorial notebook demonstrating the core ray tracing primitives:

- **Rays**: Initialize and manipulate light rays in 3-D space
- **Reflecting Planes**: Simulate mirror reflections
- **Refracting Planes**: Simulate refraction at material boundaries (e.g., air-glass interface)
- **Prism-Mirror**: Model complex prism geometries with multiple refracting/reflecting surfaces
- **Camera**: Projection and reprojection operations

This notebook provides hands-on examples for understanding the building blocks of the ray tracing system.

### 2. `runme_pytorch_calibration_test_notebook.ipynb`

A calibration pipeline for a two-camera prism setup, given camera intrinsics and approximate prism initialization parameters:

- Loads camera intrinsics, extrinsics and calibration grid data
- Initializes prism-camera arena from calibration grid data and camera intrinsics
- Trains calibration parameters using:
  - Reprojection loss (2-D pixel errors)
  - Pairwise distance constraints (3-D geometry of the calibration grid)
  - Intersection penalties (ensuring rays intersect)
- Evaluates calibration quality via reprojection and triangulation errors
- Outputs trained model checkpoints and visualization plots

**Key inputs**: Camera intrinsics (.mat files), calibration grid images, pairwise distance data

**Key outputs**: Trained model weights, reprojection error plots, 3D reconstructions

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- SciPy
- tqdm
- TensorBoard (for training visualization)

## Usage

Start with `runme_pytorch_raytracing_notebook.ipynb` to understand the ray tracing components, then proceed to `runme_pytorch_calibration_test_notebook.ipynb` for full calibration.
