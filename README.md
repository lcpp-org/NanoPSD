# NanoPSD
Software Package for Analyzing Plasma Synthesized Nano-Particle Size Distribution

# NanoPSD Pipeline

This production-ready Python project extracts particle size distributions from SEM images.
Features:
- Automated scale bar detection
- Particle segmentation using classical methods
- Size extraction and visualization
- Modular and object-oriented design

## Usage
1. Place your image in `data/raw/` as `SEM_nano_particles.png`
2. Adjust scale bar length in `main.py`
3. Run:
```bash
python main.py
```

## 🛠 Requirements

This project uses a Conda environment. Required packages:

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Scikit-image
- SciPy
- Pandas
- Pillow

## 📦 Setup Instructions

### 1. Create and Activate Conda Environment

```bash
conda create -n imglab python=3.10
conda activate imglab 
conda install -c conda-forge opencv numpy matplotlib scikit-image scipy pandas pillow
```

## To Recreate Environment 
```bash
Recreate environment using the following command: 
conda env create -f imglab_environment.yml
```

 
