import numpy as np
import cv2
import os

def load_and_prepare(img_path):  # Load and preprocess images
    img = cv2.imread(img_path) # reads the images
    if img is None:
        print(f"Could not read: {img_path}")
        return None, None, None
    
    max_size = max(img.shape[:2])  # Resize image for consistency
    scale = 700 / max_size if max_size > 700 else 1
    img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) # Convert to grayscale and apply thresholding
    smoothed = cv2.GaussianBlur(gray_img, (5, 5), 0)  # smoothen out the image and reduce noise
    binary_img = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return img_resized, binary_img, scale

def detect_coins(binary_img, scale):  # Extract circular objects using contours
    detected_objects, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular_shapes = []
    
    for obj in detected_objects:
        perimeter = cv2.arcLength(obj, True)
        area = cv2.contourArea(obj)
        
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            min_area = 500 * (scale ** 2)
            
            if 0.75 < circularity < 1.2 and area > min_area:
                circular_shapes.append(obj)
    
    return circular_shapes

def store_images(output_img, shapes, save_path, draw_type="outline"):  # Save processed images with contours and segmentation
    result_img = output_img.copy()
    
    if draw_type == "outline":
        cv2.drawContours(result_img, shapes, -1, (0, 0, 255), 2)
    elif draw_type == "mask":
        mask_layer = np.zeros(output_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_layer, shapes, -1, 255, thickness=cv2.FILLED)
        result_img = cv2.bitwise_and(output_img, output_img, mask=mask_layer)
    
    cv2.imwrite(save_path, result_img)

def batch_process(input_dir, output_dir):  # Handle multiple images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, file)
            outline_output = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_outline.jpg")
            mask_output = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_segmented.jpg")
            
            processed_img, threshold_img, scale_factor = load_and_prepare(input_path)
            if processed_img is None:
                continue
            
            detected_shapes = detect_coins(threshold_img, scale_factor)
            store_images(processed_img, detected_shapes, outline_output, draw_type="outline")
            store_images(processed_img, detected_shapes, mask_output, draw_type="mask")
            
            print(f"{file}: Objects detected = {len(detected_shapes)}")

batch_process("input", "output")
