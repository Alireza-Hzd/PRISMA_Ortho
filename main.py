# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
from pathlib import Path
import pandas as pd
from PRISMAtoolbox import PrismaData
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

    print(img1.startTime)
    t0 = "2021-04-01"
    t1 = "2021-09-01"

    #S2download.s2download(data_folder / "S2_reference_big2.tif", img1.minLon, img1.minLat, img1.maxLon, img1.maxLat, t0, t1)


"""
    
    plt.imshow(pan, vmin=np.nanpercentile(pan, 2), vmax=np.nanpercentile(pan, 98))
    plt.title('Pancromatic')
    plt.colorbar()
    plt.show()

"""
