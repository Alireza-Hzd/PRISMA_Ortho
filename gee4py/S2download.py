import ee
import requests
import shutil

# Initialize the Earth Engine module.
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


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

    # def image_poly(polygon):
    imgs = imgs.filter(ee.Filter.contains('.geo', aoi))

    def get_clouds_per(img):

        img = ee.Image(img.get('s2cloudless'))

        eo = ee.Dictionary(img.reduceRegion(ee.Reducer.median(), aoi, 100))

        return ee.Feature(None, {'prob': eo.get('probability')})

    print("Number of images", imgs.size().getInfo())

    if imgs.size().getInfo() > 0:
        # Get the best image id sorting respect to the cloud %
        results = imgs.map(get_clouds_per).sort('prob')
        best_image_id = results.first().id().getInfo()

        best_image = ee.Image("COPERNICUS/S2_SR/" + best_image_id).addBands(
            ee.Image("COPERNICUS/S2_CLOUD_PROBABILITY/" + best_image_id))

        return best_image.unmask()
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