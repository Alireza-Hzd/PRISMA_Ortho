# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
import numpy as np
import pandas as pd
import cv2
import rasterio

from matplotlib import pyplot as plt
from pathlib import Path
from PRISMAtoolbox import PrismaData
from datetime import datetime, timedelta
from gee4py import S2download

MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.15

if __name__ == '__main__':

    data_folder = Path("/content/drive/Shareddrives/GRAW_Sapienza_Team/PRISMA_IMAGES/Foggia/PRISMA_L2C")
    file_name = "PRS_L2C_STD_20220429095455_20220429095459_0001.he5"
    gcps_file_name = "/content/drive/Shareddrives/GRAW_Sapienza_Team/PRISMA_IMAGES/Foggia/vector/gcp_foggia_32633_dsmvalues/gcptest.csv"
    output_file = data_folder / (file_name.split('.')[0] + '_ortho_gcp.tif')
    dem_file = '/content/drive/Shareddrives/GRAW_Sapienza_Team/PRISMA_IMAGES/DSM/dsm_foggia_32633.tif'

    # Initialize a PrismaData object
    img1 = PrismaData(data_folder / file_name)

    # Read GCPs csv file using Pandas
    GCPs_img1 = pd.read_csv(data_folder / gcps_file_name, delimiter=";")

    # Orthorectify the Prisma Hypercube using GCPs refined RPC model
    #img1.orthorectify_hyp_cube(output_file, dem_file, GCPs_data=GCPs_img1)

'''
    # Use the GEE API to download the best cloud free Sentinel-2 image covering the PRISMA img

    t0 = img1.date + timedelta(weeks=-4)
    t1 = img1.date + timedelta(weeks=+4)

    #print(t0, t1)

    #print(img1.vnir_central_wavelengths)
    #print(img1.vnir_bands_amplitude)

    #print(img1.swir_central_wavelengths)
    #print(img1.swir_bands_amplitude)

    print(img1.L2ScaleSwirMin, img1.L2ScaleSwirMax)
    print(img1.L2ScaleVnirMin, img1.L2ScaleVnirMax)

    hypcube = img1.get_hyp_cube()

    print(hypcube.shape)

    band840 = hypcube[90, :, :]

    # plot one band
    #src_im1 = rasterio.open(data_folder / "S2_reference_test_crs_bands.tif")
    #im1 = src_im1.read(1)

    print(np.min(band840), np.max(band840), np.nanpercentile(band840, 98))

    #band840uint = (255.0 * band840 / np.nanpercentile(band840, 98)).astype('uint8')
    #im1uint = (255.0 * im1 / np.nanpercentile(im1, 98)).astype('uint8')

    plt.imshow(band840, vmin=np.nanpercentile(band840, 5), vmax=np.nanpercentile(band840, 95))
    plt.title('Band 840')
    plt.colorbar()
    plt.show()

    #plt.imshow(im1uint, vmin=np.nanpercentile(im1uint, 2), vmax=np.nanpercentile(im1uint, 98))
    #plt.title('S2 Reference')
    #plt.colorbar()
    #plt.show()

    #print(type(im1))
    #print(type(band840))




    # Convert images to grayscale
    # Convert images to grayscale
    #im1Gray = cv2.cvtColor(im1.astype('uint8'), cv2.COLOR_BGR2GRAY)

    #im2Gray = cv2.cvtColor(band840.astype('uint8'), cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    #orb = cv2.ORB_create(MAX_FEATURES)

    #kp1, dsp1 = orb.detectAndCompute(band840uint, None)
    #keypoints2, descriptors2 = orb.detectAndCompute(im1uint, None)

    #img = cv2.drawKeypoints(band840uint, keypoints1, band840)

    #plt.imshow(img)
    #plt.title('TPs detected')
    #plt.colorbar()
    #lt.show()

    #print(keypoints1[0])

    #cv2.imwrite('sift_keypoints.jpg', img)

"""
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    print(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, band840, keypoints2, matches, None)

    #S2download.s2download(data_folder / "S2_reference_test_crs_bands.tif", img1.minLon, img1.minLat, img1.maxLon,
    #                      img1.maxLat, t0, t1, crs="EPSG:32631")


    
    plt.imshow(pan, vmin=np.nanpercentile(pan, 2), vmax=np.nanpercentile(pan, 98))
    plt.title('Pancromatic')
    plt.colorbar()
    plt.show()

"""
'''
