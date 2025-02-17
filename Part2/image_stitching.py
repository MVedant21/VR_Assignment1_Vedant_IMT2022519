import numpy as np
import cv2
import os


def scale_image(img, new_width):   # Function to resize an image while keeping the aspect ratio
    h, w = img.shape[:2]
    factor = new_width / float(w)
    new_h = int(h * factor)
    return cv2.resize(img, (new_width, new_h), interpolation=cv2.INTER_AREA)

def extract_features(gray_img):   # Function to extract keypoints and descriptors
    sift = cv2.SIFT_create()    # Using SIFT to find the optimal descriptors and interest points
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    keypoints = np.float32([kp.pt for kp in keypoints])
    return keypoints, descriptors

def get_matches(kp1, kp2, desc1, desc2, ratio_thresh, reproj_thresh):   # Function to find keypoint matches between two images
    matcher = cv2.BFMatcher()
    initial_matches = matcher.knnMatch(desc1, desc2, 2)
    good_matches = []
    
    for m in initial_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio_thresh:
            good_matches.append((m[0].trainIdx, m[0].queryIdx))
    
    if len(good_matches) > 4:
        pts1 = np.float32([kp1[i] for (_, i) in good_matches])
        pts2 = np.float32([kp2[i] for (i, _) in good_matches])
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
        return good_matches, H, status
    return None

def visualize_matches(img1, img2, kp1, kp2, matches, match_status):   # Function to overlay keypoint matches between two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    output[0:h1, 0:w1] = img1
    output[0:h2, w1:] = img2
    
    for ((train_idx, query_idx), s) in zip(matches, match_status):
        if s == 1:
            pt1 = (int(kp1[query_idx][0]), int(kp1[query_idx][1]))
            pt2 = (int(kp2[train_idx][0]) + w1, int(kp2[train_idx][1]))
            cv2.line(output, pt1, pt2, (0, 255, 0), 1)
    
    return output

def refine_image(img):    # Function to crop the stitched image to remove black borders
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]
    return img

def merge_images(pair, ratio=0.75, reproj=4.0, debug=False):    # Function to stitch two images together
    right_img, left_img = pair
    
    kp1, desc1 = extract_features(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    kp2, desc2 = extract_features(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
    
    match_result = get_matches(kp1, kp2, desc1, desc2, ratio, reproj)
    if match_result is None:
        print("Insufficient matches found to stitch images together.")
        return None
    
    matches, homography, status = match_result
    
    panorama = cv2.warpPerspective(left_img, homography, 
                                   (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    panorama[0:right_img.shape[0], 0:right_img.shape[1]] = right_img
    
    panorama = refine_image(panorama)
    
    if debug:
        debug_output = visualize_matches(left_img, right_img, kp1, kp2, matches, status)
        return panorama, debug_output
    
    return panorama

def process_directory(src_folder, dest_folder):    
    file_list = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith(('.jpg', '.png'))]
    
    file_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if len(file_list) < 2:
        print(f"Not enough images to stitch in {src_folder}. At least 2 are required.")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    base_img = cv2.imread(file_list[0])
    base_img = scale_image(base_img, 600)  # Resize for consistency

    for i in range(1, len(file_list)):
        next_img = cv2.imread(file_list[i])
        next_img = scale_image(next_img, 600)

        result = merge_images([base_img, next_img], debug=True)

        if result is None:
            print(f"Skipping {file_list[i]} due to insufficient matches.")
            continue
        
        base_img, match_viz = result
        
        cv2.imwrite(os.path.join(dest_folder, f"{src_folder}_match_{i}.jpg"), match_viz)     # Save match visualization for each step

    panorama_path = os.path.join(dest_folder, f"{src_folder}_panorama.jpg")     # Save final panorama
    cv2.imwrite(panorama_path, base_img)
    
    print(f"Panorama and keypoint matches for {src_folder} created and saved in {dest_folder}.")


process_directory("input1", "output")
# process_directory("input2", "output")
