# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS

from gee4py import S2download
import rpcsat
import pandas as pd

from sklearn.model_selection import KFold

import ee
ee.Initialize()

if __name__ == '__main__':

    data_folder = Path("/Volumes/Samsung_T5/Satellite_Imagery/PRISMA")
    file_name = "PRS_L2C_STD_20200420104916_20200420104920_0001.he5"
    gcps_file_name = "GCP.csv"

    f = h5py.File(data_folder / file_name, 'r')

    print("Image Acquisition Start Time:", f.attrs.get('Product_StartTime'))
    print("UL Corner:", f.attrs.get('Product_ULcorner_lat'), f.attrs.get('Product_ULcorner_long'))
    print("LR Corner:", f.attrs.get('Product_LRcorner_lat'), f.attrs.get('Product_LRcorner_long'))

    t0 = ee.Date("2021-04-01")
    t1 = ee.Date("2021-09-01") # TO ADD INPUT VARIABLE and MOVE IN S2 download function

    minLat, maxLat = float(f.attrs.get('Product_LRcorner_lat')), float(f.attrs.get('Product_ULcorner_lat'))
    minLon, maxLon = float(f.attrs.get('Product_LRcorner_long')),float(f.attrs.get('Product_ULcorner_long'))

    print(minLon, minLat, maxLon, maxLat)
    #S2download.s2download(data_folder / "S2_reference_big2.tif", minLon, minLat, maxLon, maxLat, t0, t1)

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

    # HERE PROVIDE THE CSV FILE WITH THE GCPs TO REFINE THE RPCs
    GCPs_data = pd.read_csv(data_folder / gcps_file_name, delimiter=";")
    print(GCPs_data.head())
    RPC_Pan.GCP_assessment(GCPs_data)
    #RPC_Pan.GCP_refinement(GCPs_data)
    #RPC_Pan.GCP_assessment(GCPs_data)
    N_GCP = GCPs_data.shape[0]

    print("")
    print("--- Leave one out (K-fold) validation ---")

    kf = KFold(n_splits=N_GCP)
    kf.get_n_splits(GCPs_data)
    results_df = pd.DataFrame()

    for train_index, test_index in kf.split(GCPs_data):
        X_train, X_test = GCPs_data.iloc[train_index], GCPs_data.iloc[test_index]
        rpc_img_n = rpcsat.RPCmodel()
        rpc_img_n.read_from_PRISMA_h5(data_folder / file_name)
        rpc_img_n.GCP_refinement(X_train.reset_index(), verbose=False)
        Row_mean, Row_std, Col_mean, Col_std = rpc_img_n.GCP_assessment(X_test.reset_index(), verbose=False)
        X_test_copy = X_test.copy()
        X_test_copy['D_Col'] = [Col_mean]
        X_test_copy['D_Row'] = [Row_mean]
        X_test_copy['Mod'] = [np.sqrt(Row_mean * Row_mean + Col_mean * Col_mean)]
        #results_df = results_df.append(X_test_copy)
        results_df = pd.concat([results_df, X_test_copy], ignore_index=True)

    print(results_df.head(15))
    print("")
    print("--- Leave One Out Residual errors N_GCP:", N_GCP, " ----")
    print("Row_mean:", results_df[['D_Row']].mean(axis=0).to_numpy()[0], "Row_std:",
          results_df[['D_Row']].std(axis=0).to_numpy()[0])
    print("Col_mean:", results_df[['D_Col']].mean(axis=0).to_numpy()[0], "Col_std:",
          results_df[['D_Col']].std(axis=0).to_numpy()[0])

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