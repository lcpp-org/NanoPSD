import os
from pipeline.analyzer import NanoparticleAnalyzer

os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Define path to input SEM image and known scale bar length in nanometers
# image_path = "data/raw/SEM_nano_particles.png"
image_path = "./SEM_nano_particles.png"
scale_bar_length_nm = 100  # known physical length

# Instantiate analyzer and run the full pipeline
analyzer = NanoparticleAnalyzer(image_path, scale_bar_length_nm)
analyzer.run()
