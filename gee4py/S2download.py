import ee
import requests
import shutil

# Initialize the Earth Engine module.
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


# Function to mosaic by date, orbit, etc
def mosaicBy(imcol):
    # imcol: An image collection
    # returns: An image collection
    # return the collection as a list of images (not an image collection)

    imlist = imcol.toList(imcol.size());

    # Get all the dates as list
    #def fun1(im):
    #    return ee.Image(im).date().format("YYYY-MM-dd");

    all_dates = imlist.map(lambda im: ee.Image(im).date().format("YYYY-MM-dd"))

    def fun2(im):
        return ee.Image(im).get('SENSING_ORBIT_NUMBER')

    all_orbits = imlist.map(fun2)

    def fun3(im):
        return ee.Image(im).get('SPACECRAFT_NAME')

    all_spNames = imlist.map(fun3)

    concat_all = all_dates.zip(all_orbits).zip(all_spNames)

    def fun4(el):
        return ee.List(el).flatten().join(" ")

    concat_all = concat_all.map(fun4)

    concat_unique = concat_all.distinct()

    def fun5(d):
        d1 = ee.String(d).split(" ")
        date1 = ee.Date(d1.get(0));
        orbit = ee.Number.parse(d1.get(1)).toInt();
        spName = ee.String(d1.get(2));
        im = imcol.filterDate(date1, date1.advance(1, "day")) \
            .filterMetadata('SPACECRAFT_NAME', 'equals', spName) \
            .filterMetadata('SENSING_ORBIT_NUMBER', 'equals', orbit) \
            .mosaic()

        im = im.copyProperties(imcol.first())

        return im.set("system:time_start", date1.millis(), "system:date", date1.format("YYYY-MM-dd"), "system:id", d1)

    mosaic_imlist = concat_unique.map(fun5)

    return ee.ImageCollection(mosaic_imlist)


def get_best_s2_image(aoi, start_date, end_date):
    # Import and filter S2 SR.

    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date))

    print("Number of S2 images", s2_sr_col.size().getInfo())

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    imgs = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    # add the probability mask from s2cloudless to the s2 bands
    def add_cloud_probability(img):
        cloud_mask = ee.Image(img.get('s2cloudless'))
        return img.addBands(cloud_mask)

    imgs = imgs.map(add_cloud_probability)

    print("Number of S2 Granules", imgs.size().getInfo())

    s2day = mosaicBy(imgs)

    print("Number of S2 acquisitions", s2day.size().getInfo())

    imgs = s2day

    # Function to count the number of cloud free pixel with the reference AOI -> assume 50% th of cloud probability
    def get_clouds_per(img):

        cloud_mask = ee.Image(img.get('s2cloudless'))

        eo = ee.Dictionary(cloud_mask.reduceRegion(ee.Reducer.fixedHistogram(0, 100, 2), aoi, 100))

        cloudfree_px = ee.Array(ee.Array(eo.get('probability')).toList().get(0)).toList().get(1)

        return img.set("prob", cloudfree_px)

    if imgs.size().getInfo() > 0:
        # Get the best image id sorting respect to the cloud %
        results = imgs.map(get_clouds_per).sort('prob', False)
        return results.first().unmask()
    else:
        return None


def s2download(filename, xmin, ymin, xmax, ymax, t0, t1):
    date_t0 = ee.Date(t0)
    data_t1 = ee.Date(t1)
    # Generate the desired image from the given reference tile.
    # point = ee.Geometry.Point(p)
    # region = point.buffer(127.8*10).bounds(maxError=0.1, proj=CRS)

    region = ee.Geometry.Rectangle(xmin, ymin, xmax, ymax) #list(point[1]['geometry'].exterior.coords)).buffer(-13.0, proj=CRS)

    # image = (dataset
    #         .filterBounds(region)
    #        .filterDate(date, date.advance(1,'month'))
    #         .select(inBands)
    #         .median()
    #        .clip(region))

    image = get_best_s2_image(region, date_t0, data_t1)

    CRS = "EPSG:4326"
    # S2 relevant bands
    inBands = ["B8"]

    if image != None:
        image = image.select(inBands).clip(region)

        if len(image.bandNames().getInfo()) > 0:

            # Fetch the URL from which to download the image.
            url = image.getDownloadURL({'scale': 10, 'region': region, 'format': "GEO_TIFF", 'crs': CRS})
            # url = image.getThumbURL({
            ##    'dimensions': '256x256',
            #   'format': 'jpg'})

            # Handle downloading the actual pixels.
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()

            #filename = 'img_%05d_S2_%s_%02d.tif' % (point[0] + N0, point[1]['year'], month)
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            print("Done")
        else:
            print('Missing Tile')





