import cv2
import os
import numpy as np

def preprocess_image(image_path: str, output_dir: str, size: tuple = (256, 256)):
    """
    Resizes an image and converts it to RGB and HSV color spaces.

    Args:
        image_path (str): The path to the input image.
        output_dir (str): Directory to save the processed image.
        size (tuple): The target size for the image (width, height).

    Returns:
        A tuple containing the processed image in RGB and HSV formats.
        Returns (None, None) if the image cannot be read.
    """
    try:
        with open(image_path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            image_bgr = cv2.imdecode(chunk, cv2.IMREAD_COLOR)

        if image_bgr is None:
            print(f"CRITICAL: OpenCV failed to decode {image_path}")
            return None, None

        # Resize the image
        image_resized_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)

        # Convert to RGB and HSV
        image_rgb = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2HSV)

        # Save the processed image (as RGB)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        # Convert RGB back to BGR for saving with OpenCV
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        return image_rgb, image_hsv
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None