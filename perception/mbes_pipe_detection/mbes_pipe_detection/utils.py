"""
Utility functions for the MBES pipe detection module.
"""

import cv2
import numpy as np
from scipy.interpolate import griddata
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_bilateral

def pcl_buffer_to_intensity(pcl_buffer, resolution):
    """
    Convert pcl buffer of shape (num_pings, num_points, 4) to an intensity image.
    The intensity image will have shape (num_y_pixels, num_x_pixels) where
    num_x_pixels and num_y_pixels are determined by the resolution and the range of x and y coordinates.

    Returns:
        dict: A dictionary containing:
            - 'intensity_image': The interpolated intensity image.
            - 'mask': A boolean mask indicating valid pixels in the intensity image.
        if pcl_buffer is None or has insufficient data, returns None.
    """
    if pcl_buffer is None:
        return None
    x = pcl_buffer[:, :, 0]
    y = pcl_buffer[:, :, 1]
    z = pcl_buffer[:, :, 2]
    intensities = pcl_buffer[:, :, 3]

    mean_intensity = np.mean(intensities, axis=0)
    intensities /= mean_intensity

    x_min, x_max = (np.min(x), np.max(x))
    y_min, y_max = (np.min(y), np.max(y))
    num_x_pixels = int((x_max - x_min) / resolution)
    num_y_pixels = int((y_max - y_min) / resolution)

    if num_x_pixels <= 0 or num_y_pixels <= 0:
        return None

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, num_x_pixels),
        np.linspace(y_min, y_max, num_y_pixels)
    )
    intensity_image = griddata(
        (x.flatten(), y.flatten()),
        intensities.flatten(),
        (X, Y),
        method='linear',
    )
    Z = griddata(
        (x.flatten(), y.flatten()),
        z.flatten(),
        (X, Y),
        method='linear',
    )

    gradient_image = np.gradient(intensity_image, axis=1)

    mask = ~np.isnan(gradient_image)    # used to be mask = ~np.isnan(intensity_image) 
    return {
        'x': X,
        'y': Y,
        'z': Z,
        'intensity_image': intensity_image,
        'gradient_image': gradient_image,
        'mask': mask,
    }


def normalize_image(intensity_image):
    """
    Normalize the intensity image to the range [0, 255].
    """
    normalized_image = cv2.normalize(
        intensity_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return normalized_image

def img_to_uint8(image):
    """
    Convert an image to uint8 format.
    If the image is already in uint8 format, it returns the image unchanged.
    """
    if image.dtype == np.uint8:
        return image
    else:
        image = np.nan_to_num(image, nan=0)  
        return image.astype(np.uint8)

def process_intensity_image(intensity_dict):
    intensity_image = intensity_dict['intensity_image']
    intensity_image = np.where(np.isnan(intensity_image), np.nanmean(intensity_image), intensity_image)  # replace NaNs with 0
    intensity_image = denoise_bilateral(
        intensity_image,
        sigma_color=0.3,
        sigma_spatial=5,
    )
    intensity_image = ((intensity_image - np.min(intensity_image)) / (np.max(intensity_image) - np.min(intensity_image)) * 255).astype(np.uint8)
    edges = cv2.Canny(intensity_image, 100, 150)
    mask = intensity_dict['mask']
    edges = np.logical_and(edges, mask)  # apply mask to edges

    lines = cv2.HoughLinesP(edges.astype(np.uint8), 1, np.pi / 180,
                            threshold=10,
                            minLineLength=50,
                            maxLineGap=20)
    return lines

def draw_lines_on_image(image, lines):
    """
    Draw lines on the given image.

    Parameters:
    - image: 2D numpy array representing the image.
    - lines: list of lines to draw, where each line is represented as a tuple (x1, y1, x2, y2).

    Returns:
    - image_with_lines: The input image with lines drawn on it.
    """
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 16, 240), 1, cv2.LINE_AA)
    return image_with_lines



def pipeline_detect(grad_img, mask=None, min_frac_in_mask=0.8):
    """
    Detect pipelines in the given data using Hough Transform.
    
    Parameters:
    - grad_img: 2D numpy array representing the gradient image of the intensity; should already be a unit8 type image. 
    - mask: optional boolean mask to filter the image; 
    - min_frac_in_mask: minimum fraction of pixels in the mask that must be part of a detected line to consider it a valid pipeline, otherwise we assume it is the edge of the image that has been detected as a pipeline.
    
    Returns:
    - pipeline: boolean indicating if a pipeline was detected.
    - mid_x: x-coordinate of the midpoint of the detected pipeline.
    - mid_y: y-coordinate of the midpoint of the detected pipeline.
    """

    # make Hough lines but the probabilistic kind
    # linesP = cv2.HoughLinesP(grad_img, 1, np.pi / 180, 20, None, 20, 50)
    linesP = cv2.HoughLinesP(grad_img, 1, np.pi / 180,
                            threshold=10,
                            minLineLength=50,
                            maxLineGap=20)


    pipeline = False

    # we collect all the x and y coordinates of the lines to determine the midpoint coordinate of where the pipeline probably is.
    all_x = []
    all_y = []

    # draw the lines on the image 
    # we assume if there are lines, then there is a pipeline
    if linesP is not None:
        valid_lines = 0
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1, y1, x2, y2 = l

            # Sample points along the line
            num_points = int(np.hypot(x2 - x1, y2 - y1))
            xs = np.linspace(x1, x2, num_points).astype(int)
            ys = np.linspace(y1, y2, num_points).astype(int)

            # Check mask coverage
            if mask is not None and num_points > 0:
                xs_clipped = np.clip(xs, 0, mask.shape[1] - 1)
                ys_clipped = np.clip(ys, 0, mask.shape[0] - 1)
                mask_values = mask[ys_clipped, xs_clipped]
                fraction_in_mask = np.count_nonzero(mask_values) / num_points
                if fraction_in_mask < min_frac_in_mask:
                    continue  # Discard this line

            # If we get here, the line is valid
            valid_lines += 1
            cv2.line(grad_img, (x1, y1), (x2, y2), (255,16,240), 1, cv2.LINE_AA)
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        if valid_lines > 0:
            pipeline = True

    mid_x = np.mean(all_x) if all_x else None
    mid_y = np.mean(all_y) if all_y else None

    return pipeline, mid_x, mid_y, grad_img
