# Import necessary modules
import logging
import os
from glob import glob

# Import project-specific modules from their respective paths
from utils.scale_bar import detect_scale_bar_length  # Detects scale bar in pixels
from scripts.preprocessing.clahe_filter import preprocess_image  # Enhances image using CLAHE and other filters
from scripts.segmentation.otsu_segment import segment_particles  # Performs Otsu thresholding for segmentation
from scripts.analysis.size_measurement import measure_particles, export_to_latex  # Measures particle sizes and exports
from scripts.visualization.plotting import plot_results  # Plots size distribution results

# Set up basic logging configuration
# This will show timestamps, logging levels, and messages in console
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class NanoparticleAnalyzer:
    """
    A class for analyzing nanoparticle sizes from microscopy images.
    
    Parameters:
    -----------
    image_path : str
        File path to a single image or directory containing multiple images.
    scale_bar_nm : float
        Real-world length of the scale bar in nanometers (e.g., 100 for 100 nm).
    batch : bool
        If True, processes all .png images in a folder; else, processes a single image.
    """

    def __init__(self, image_path, scale_bar_nm, batch=False):
        self.image_path = image_path          # Path to image or directory
        self.scale_bar_nm = scale_bar_nm      # Real length of the scale bar in nanometers
        self.batch_mode = batch               # Whether to process multiple images

    def run(self):
        """
        Main entry point for the analysis.
        Determines whether to run in batch mode or single-image mode.
        """
        if self.batch_mode:
            # Collect all .png images in the specified folder
            images = glob(os.path.join(self.image_path, "*.png"))
            for img in images:
                try:
                    self._process_image(img)
                except Exception as e:
                    # Log error if any image fails to process
                    logging.error(f"Failed to process {img}: {e}")
        else:
            # Process only the single image provided
            self._process_image(self.image_path)
    
    def _process_image(self, img_path):
        """
        Process an individual image: calibrate scale, segment particles, measure size, and visualize results.

        Parameters:
        -----------
        img_path : str
            Full file path to the image to be processed.
        """
        logging.info(f"Processing: {img_path}")

        # Step 1: Detect scale bar length in pixels from the image
        scale_bar_px, _ = detect_scale_bar_length(img_path)

        # Step 2: Compute calibration ratio: nanometers per pixel
        nm_per_pixel = self.scale_bar_nm / scale_bar_px
        logging.info(f"Calibration: {nm_per_pixel:.2f} nm/pixel")

        # Step 3: Preprocess the image (e.g., contrast enhancement using CLAHE)
        binary, original = preprocess_image(img_path)

        # Step 4: Segment particles using Otsu's thresholding and return labeled regions
        labeled_image, regions = segment_particles(binary)

        # Step 5: Measure diameters of segmented particles and convert to nanometers
        diameters_nm, _, _ = measure_particles(regions, labeled_image, original, nm_per_pixel)

        # Step 6: Visualize results - plot histogram of particle sizes
        plot_results(diameters_nm, img_path)

        # Step 7: Export measurements to LaTeX-friendly format (e.g., table)
        export_to_latex(diameters_nm, img_path)

        logging.info(f"Completed: {len(diameters_nm)} particles analyzed")