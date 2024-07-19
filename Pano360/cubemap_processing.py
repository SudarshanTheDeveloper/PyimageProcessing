import cv2
import numpy as np
import os

# Define the base directory
base_dir = "D:/Py_Projects/Pano360"

# Define the input and output directories
input_dir = os.path.join(base_dir, "D:/Py_Projects/Pano360/input_images")
output_dir = os.path.join(base_dir, "D:/Py_Projects/Pano360/output_images")

# Load the six images
images = {
    "right": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/right.jpg")),
    "left": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/left.jpg")),
    "top": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/top.jpg")),
    "bottom": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/bottom.jpg")),
    "front": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/front.jpg")),
    "back": cv2.imread(os.path.join(input_dir, "D:/Py_Projects/Pano360/back.jpg"))
}

# Function to align images
def align_images(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    return aligned_img

# Align the front image with the right image
images["front"] = align_images(images["front"], images["right"])

# Smooth the edges of each image
smoothed_images = {}
for key, img in images.items():
    # Apply Gaussian blur to the edges
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    smoothed_images[key] = blurred

# Combine the smoothed images into a single cubemap image
size = images["right"].shape[0]  # Assuming all images are square and of the same size
cubemap = np.zeros((3 * size, 4 * size, 3), dtype=np.uint8)

cubemap[size:2*size, size:2*size] = smoothed_images["right"]  # right
cubemap[size:2*size, 3*size:4*size] = smoothed_images["left"]  # left
cubemap[0:size, size:2*size] = smoothed_images["top"]  # top
cubemap[2*size:3*size, size:2*size] = smoothed_images["bottom"]  # bottom
cubemap[size:2*size, 2*size:3*size] = smoothed_images["front"]  # front
cubemap[size:2*size, 0:size] = smoothed_images["back"]  # back

# Display or save the cubemap image
cv2.imshow("Cubemap", cubemap)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, "cubemap_smoothed.jpg"), cubemap)  # Save the cubemap image
