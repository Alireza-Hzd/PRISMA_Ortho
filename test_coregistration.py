import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


import rasterio






data_folder = "/Volumes/Samsung_T5/Satellite_Imagery/PRISMA/"

# Read the images to be aligned
src_im1 = rasterio.open(data_folder + "S2_referencealigned.tif")
src_im2 = rasterio.open(data_folder + "PRS_L2C_STD_20200420104916_20200420104920_0001aligned.tif")

im1 = src_im1.read(1)
im2 = src_im2.read(1)

# Convert images to grayscale
im1_gray = cv2.convertScaleAbs(im1, alpha=(255.0/np.percentile(im1, 95)))
im2_gray = cv2.convertScaleAbs(im2, alpha=(255.0/np.percentile(im2, 95)))

# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_TRANSLATION

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else:
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 100;

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
print("done")
print(warp_matrix)

if warp_mode == cv2.MOTION_HOMOGRAPHY:
# Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
# Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

# Show final results
plt.imshow(im1)
plt.imshow(im2)
plt.imshow(im2_aligned)
plt.show()

with rasterio.Env():

    # Write an array as a raster band to a new 8-bit file. For
    # the new file's profile, we start with the profile of the source
    profile = src_im2.profile

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    #profile.update(
    #    dtype=rasterio.uint8,
    #    count=1,
    #    compress='lzw')

    with rasterio.open(data_folder + 'testcoregistration.tif', 'w', **profile) as dst:
        dst.write(im2_aligned, 1)
