import os, sys, time
import numpy as np
from osgeo import ogr, gdal, gdalconst
from osgeo import gdal_array as ga

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def stretch_n(bands, img_min, img_max, lower_percent=0, higher_percent=100):
    """
    :param bands:  目标数据，numpy格式
    :param img_min:   目标位深的最小值，以8bit为例，最大值为255， 最小值为0
    :param img_max:    目标位深的最大值
    :return:
    """
    out = np.zeros_like(bands).astype(np.float32)
    a = img_min
    b = img_max
    c = np.percentile(bands[:, :], lower_percent)
    d = np.percentile(bands[:, :], higher_percent)
    t = a + (bands[:, :] - c) * (b - a) / (d - c)
    t[t < a] = a
    t[t > b] = b
    out[:, :] = t
    return out


def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW'])

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset


def image_resampling(source_file, target_file, scale=5.):
    """
          image resampling
    :param source_file: the path of source file
    :param target_file: the path of target file
    :param scale: pixel scaling
    :return: None
    """
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    band_count = dataset.RasterCount  # 波段数

    if band_count == 0 or not scale > 0:
        print("参数异常")
        return

    cols = dataset.RasterXSize  # 列数
    rows = dataset.RasterYSize  # 行数
    cols = int(cols * scale)  # 计算新的行列数
    rows = int(rows * scale)

    geotrans = list(dataset.GetGeoTransform())
    print(dataset.GetGeoTransform())
    print(geotrans)
    geotrans[1] = geotrans[1] / scale  # 像元宽度变为原来的scale倍
    geotrans[5] = geotrans[5] / scale  # 像元高度变为原来的scale倍
    print(geotrans)

    if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
        os.remove(target_file)  # 则删除之

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    target.SetProjection(dataset.GetProjection())  # 设置投影坐标
    target.SetGeoTransform(geotrans)  # 设置地理变换参数
    total = band_count + 1
    for index in range(1, total):
        # 读取波段数据
        print("正在写入" + str(index) + "波段")
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
        out_band = target.GetRasterBand(index)
        # out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(data)  # 写入数据到新影像中
        out_band.FlushCache()
        out_band.ComputeBandStats(False)  # 计算统计信息
    print("正在写入完成")
    del dataset


def sample_clip(shp, tif, outputdir, sampletype, size, fieldName='cls', n=None):
    """
        according to sampling point, generating image slices
    :param shp: the path of shape file
    :param tif: the path of image
    :param outputdir: the directory of output
    :param sampletype: line or polygon
    :param size:  the size of images slices
    :param fieldName: the name of field
    :param n: the start number
    :return:
    """
    time1 = time.clock()

    gdal.AllRegister()
    lc = gdal.Open(tif)
    im_width = lc.RasterXSize
    im_height = lc.RasterYSize
    im_geotrans = lc.GetGeoTransform()
    bandscount = lc.RasterCount
    im_proj = lc.GetProjection()
    print(im_width, im_height)
    gdal.AllRegister()
    gdal.SetConfigOption("gdal_FILENAME_IS_UTF8", "YES")

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dsshp = driver.Open(shp, 0)
    if dsshp is None:
        print('Could not open ' + 'sites.shp')
        sys.exit(1)
    layer = dsshp.GetLayer()
    xValues = []
    yValues = []
    m = layer.GetFeatureCount()
    feature = layer.GetNextFeature()
    print("tif_bands:{0},samples_nums:{1},sample_type:{2},sample_size:{3}*{3}".format(bandscount, m, sampletype,
                                                                                      int(size)))

    if n is not None:
        pass
    else:
        n = 1
    while feature:
        if n < 10:
            dirname = "0000000" + str(n)
        elif n >= 10 and n < 100:
            dirname = "000000" + str(n)
        elif n >= 100 and n > 1000:
            dirname = "00000" + str(n)
        else:
            dirname = "0000" + str(n)

        # print dirname
        dirpath = os.path.join(outputdir, dirname + "_V1")
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        tifname = dirname + ".tif"
        if "poly" in sampletype or "POLY" in sampletype:
            shpname = dirname + "_V1_POLY.shp"
        if "line" in sampletype or "LINE" in sampletype:
            shpname = dirname + "_V1_LINE.shp"
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        print(x, y)
        print(im_geotrans)
        xValues.append(x)
        yValues.append(y)
        newform = []
        newform = list(im_geotrans)
        # print newform
        newform[0] = x - im_geotrans[1] * int(size) / 2.0
        newform[3] = y - im_geotrans[5] * int(size) / 2.0
        print(newform[0], newform[3])
        newformtuple = tuple(newform)
        x1 = x - int(size) / 2 * im_geotrans[1]
        y1 = y - int(size) / 2 * im_geotrans[5]
        x2 = x + int(size) / 2 * im_geotrans[1]
        y2 = y - int(size) / 2 * im_geotrans[5]
        x3 = x - int(size) / 2 * im_geotrans[1]
        y3 = y + int(size) / 2 * im_geotrans[5]
        x4 = x + int(size) / 2 * im_geotrans[1]
        y4 = y + int(size) / 2 * im_geotrans[5]
        Xpix = (x1 - im_geotrans[0]) / im_geotrans[1]
        # Xpix=(newform[0]-im_geotrans[0])

        Ypix = (newform[3] - im_geotrans[3]) / im_geotrans[5]
        # Ypix=abs(newform[3]-im_geotrans[3])
        print("#################")
        print(Xpix, Ypix)

        # **************create tif**********************
        # print"start creating {0}".format(tifname)
        pBuf = None
        pBuf = lc.ReadAsArray(int(Xpix), int(Ypix), int(size), int(size))
        # print pBuf.dtype.name
        driver = gdal.GetDriverByName("GTiff")
        create_option = []
        if 'int8' in pBuf.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in pBuf.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        outtif = os.path.join(dirpath, tifname)
        ds = driver.Create(outtif, int(size), int(size), int(bandscount), datatype, options=create_option)
        if ds == None:
            print("2222")
        ds.SetProjection(im_proj)
        ds.SetGeoTransform(newformtuple)
        ds.FlushCache()
        if bandscount > 1:
            for i in range(int(bandscount)):
                outBand = ds.GetRasterBand(i + 1)
                outBand.WriteArray(pBuf[i])
        else:
            outBand = ds.GetRasterBand(1)
            outBand.WriteArray(pBuf)
        ds.FlushCache()
        # print "creating {0} successfully".format(tifname)
        # **************create shp**********************
        # print"start creating shps"
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        strVectorFile = os.path.join(dirpath, shpname)
        ogr.RegisterAll()
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(shp)
        layer0 = ds.GetLayerByIndex(0)
        prosrs = layer0.GetSpatialRef()
        # geosrs = osr.SpatialReference()

        oDriver = ogr.GetDriverByName("ESRI Shapefile")
        if oDriver == None:
            print("1")
            return

        oDS = oDriver.CreateDataSource(strVectorFile)
        if oDS == None:
            print("2")
            return

        papszLCO = []
        if "line" in sampletype or "LINE" in sampletype:
            oLayer = oDS.CreateLayer("TestPolygon", prosrs, ogr.wkbLineString, papszLCO)
        if "poly" in sampletype or "POLY" in sampletype:
            oLayer = oDS.CreateLayer("TestPolygon", prosrs, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("3")
            return

        oFieldName = ogr.FieldDefn(fieldName, ogr.OFTString)
        oFieldName.SetWidth(50)
        oLayer.CreateField(oFieldName, 1)
        oDefn = oLayer.GetLayerDefn()
        oFeatureRectangle = ogr.Feature(oDefn)

        geomRectangle = ogr.CreateGeometryFromWkt(
            "POLYGON (({0} {1},{2} {3},{4} {5},{6} {7},{0} {1}))".format(x1, y1, x2, y2, x4, y4, x3, y3))
        oFeatureRectangle.SetGeometry(geomRectangle)
        oLayer.CreateFeature(oFeatureRectangle)
        print("{0} ok".format(dirname))
        n = n + 1
        feature = layer.GetNextFeature()
    time2 = time.clock()
    print('Process Running time: %s min' % ((time2 - time1) / 60))

    return n