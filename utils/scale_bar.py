# Import OpenCV for image processing and NumPy for numerical operations
import cv2
import numpy as np

def detect_scale_bar_length(image_path):
    """
    Detects the horizontal length of the scale bar in a grayscale microscopy image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file.
    
    Returns:
    --------
    max_width : int
        The width (in pixels) of the detected scale bar.
    
    Raises:
    -------
    ValueError if the scale bar cannot be detected.
    """

    # Step 1: Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Get image dimensions
    height, width = img.shape  # height = number of rows, width = number of columns

    # Step 3: Define a Region of Interest (ROI) near the bottom-left corner of the image
    # Assuming the scale bar is typically located in the bottom-left corner.
    # ROI: rows from (height-50) to (height-5), columns from 5 to 150
    roi = img[height - 50:height - 5, 5:150]

    # Step 4: Apply binary thresholding to convert ROI to black and white
    # Pixels > 50 are set to 255 (white), others to 0 (black)
    _, binary = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)

    # Step 5: Invert the binary image to make the scale bar white on a black background
    # This makes it easier to detect using contours
    inverted = cv2.bitwise_not(binary)
   
    # Step 6: Find external contours in the inverted binary ROI
    # cv2.RETR_EXTERNAL returns only the outermost contours
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Initialize variable to track the maximum width found
    max_width = 0

    # Step 8: Initialize scal bar contour to save the contour
    scale_bar_contour = None

    # Step 9: Loop through each detected contour
    for cnt in contours:
        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Heuristic filter:
        #  h > 0 ensures valid height
        #  w/h > 5 ensures a long, thin horizontal shape (likely a scale bar)
        #  w > max_width ensures we keep the longest one
        if h > 0 and w / h > 5 and w > max_width:
            max_width = w  # Update maximum width
            scale_bar_contour = cnt

    # Step 10: Raise an error if no valid scale bar was found
    if max_width == 0:
        raise ValueError("Scale bar not detected.")

    # Step 10: Return the detected scale bar length in pixels
    return max_width, scale_bar_contour
