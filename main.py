# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
from pathlib import Path
import pandas as pd
from PRISMAtoolbox import PrismaData
from datetime import datetime, timedelta
from gee4py import S2download


if __name__ == '__main__':

    data_folder = Path("/Volumes/Samsung_T5/Satellite_Imagery/PRISMA")
    file_name = "PRS_L2C_STD_20200420104916_20200420104920_0001.he5"
    gcps_file_name = "GCP.csv"
    output_file = data_folder / (file_name.split('.')[0] + '_ortho_gcp.tif')
    dem_file = '/Volumes/Samsung_T5/Satellite_Imagery/PRISMA/dsm_totale.tif'

    # Initialize a PrismaData object
    img1 = PrismaData(data_folder / file_name)

    # Read GCPs csv file using Pandas
    GCPs_img1 = pd.read_csv(data_folder / gcps_file_name, delimiter=";")

    # Orthorectify the Prisma Hypercube using GCPs refined RPC model
    #img1.orthorectify_hyp_cube(output_file, dem_file, GCPs_data=GCPs_img1)


    # Use the GEE API to download the best cloud free Sentinel-2 image covering the PRISMA img

    t0 = img1.date + timedelta(weeks=-4)
    t1 = img1.date + timedelta(weeks=+4)

    print(t0, t1)

    print(img1.vnir_central_wavelengths)
    print(img1.vnir_bands_amplitude)

    print(img1.swir_central_wavelengths)
    print(img1.swir_bands_amplitude)

    #S2download.s2download(data_folder / "S2_reference_test_crs_bands.tif", img1.minLon, img1.minLat, img1.maxLon,
    #                      img1.maxLat, t0, t1, crs="EPSG:32631")

"""
    
    plt.imshow(pan, vmin=np.nanpercentile(pan, 2), vmax=np.nanpercentile(pan, 98))
    plt.title('Pancromatic')
    plt.colorbar()
    plt.show()

"""
