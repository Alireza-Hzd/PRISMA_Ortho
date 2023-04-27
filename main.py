# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS

from gee4py import S2download
import rpcsat

import ee

ee.Initialize()

if __name__ == '__main__':

    data_folder = Path("/Volumes/Samsung_T5/Satellite_Imagery/PRISMA")
    file_name = "PRS_L2C_STD_20200420104916_20200420104920_0001.he5"

    f = h5py.File(data_folder / file_name, 'r')

    print("Image Acquisition Start Time:", f.attrs.get('Product_StartTime'))
    print("UL Corner:", f.attrs.get('Product_ULcorner_lat'), f.attrs.get('Product_ULcorner_long'))
    print("LR Corner:", f.attrs.get('Product_LRcorner_lat'), f.attrs.get('Product_LRcorner_long'))

    t0 = ee.Date("2021-04-01")
    t1 = ee.Date("2021-09-01")

    minLat, maxLat = float(f.attrs.get('Product_LRcorner_lat')), float(f.attrs.get('Product_ULcorner_lat'))
    minLon, maxLon = float(f.attrs.get('Product_LRcorner_long')),float(f.attrs.get('Product_ULcorner_long'))

    minLattest = 5.90
    print(minLon, minLat, maxLon, maxLat)
    #S2download.s2download(data_folder / "S2_reference2.tif", 5.90, 50.25, 6.15, 50.35, t0, t1)
    S2download.s2download(data_folder / "S2_reference_big.tif", minLon, minLat, maxLon, maxLat, t0, t1)

    #import sys
    #sys.exit()
    pan = f['HDFEOS']['SWATHS']['PRS_L2C_PCO']['Data Fields']['Cube']

    #plt.imshow(pan, vmin=np.nanpercentile(pan, 2), vmax=np.nanpercentile(pan, 98))
    #plt.title('Pancromatic')
    #plt.colorbar()
    #plt.show()

    #rpc = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geocoding Model']
    #coeff = rpc.attrs.get('SAMP_DEN_COEFF')
    #print(coeff)
    #print(coeff.type())

    RPC_Pan = rpcsat.RPCmodel()
    RPC_Pan.read_from_PRISMA_h5(data_folder / file_name)
    RPC_Pan.write_RPC("test_rpc_prisma_pan.txt")

    xsize = pan.shape[1]
    ysize = pan.shape[0]

    rpc_dict = RPC_Pan.to_geotiff_dict()

    with rasterio.open(
        file_name.split('.')[0]+"_pan.tif", "w",
        driver="GTiff",
        height=pan.shape[0],
        width=pan.shape[1],
        count=1,
        dtype=pan.dtype,
        crs=4326
    ) as dst:
        dst.update_tags(ns="RPC", **rpc_dict)
        dst.write(pan, 1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# gdalwarp -r near -rpc -to 'RPC_DEM=dsm_totale.tif' -of GTiff PRS_L2C_STD_20200420104916_20200420104920_0001_pan.tif orthorectified_image.tif