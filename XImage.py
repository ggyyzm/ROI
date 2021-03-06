from osgeo import gdal, gdal_array
import numpy as np


def GetSuffix(filepath):
    i = filepath.rfind('.')
    string = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit() or len(filepath) - i > 5:
        return string
    else:
        string = filepath[i + 1:len(filepath)]
        return string


def GetFilePath(filepath):
    i = filepath.rfind('.')
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit():
        return filepath
    else:
        string = filepath[0:i]
        return string


class CXImage:
    """
        m_nBands：波段数
        m_nLines：height
        m_nSamples：width
        m_nDataType：数据类型
        m_strImgPath：图像路径
        currentHeight：当前窗口高度
        currentWidth：当前窗口宽度
        partWidth：分块宽度
        partHeight：分块高度
        proDataset：保存图像信息
        currentPosX：当前X坐标
        currentPosY：当前Y坐标
        data_arrange == 0 <==> (height, width, band)
                     == 1 <==> (band, height, width)(gdal)
    """

    currentHeight = 0
    currentWidth = 0
    partWidth = 0
    partHeight = 0
    currentPosX = 0
    currentPosY = 0

    # 创建对象时的初始化操作
    def __init__(self, nBands=0, nLines=0, nSamples=0, nDataType=np.float32, strImgPath=None):
        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_strImgPath = strImgPath

        self.minBandValue = None
        self.maxBandValue = None
        self.isNormalization = False
        self.proDataset = None

    # 注册图像驱动，同时设置输出图像数据类型
    def Create(self, nBands, nLines, nSamples, nDataType, strImgPath):
        """
            np.uint8 <==> gdal.GDT_Byte == 1
            np.int8 <==> gdal.GDT_Byte == 1
            np.byte <==> gdal.GDT_Byte == 1
            np.uint16 <==> gdal.GDT_UInt16 == 2
            np.int16 <==> gdal.GDT_Int16 == 3
            np.uint <==> gdal.GDT_UInt32 == 4
            np.uint32 <==> gdal.GDT_UInt32 == 4
            np.int32 <==> gdal.GDT_Int32 == 5
            np.float32 <==> gdal.GDT_Float32 == 6
            np.float64 <==> gdal.GDT_Float64 == 7
            np.complex64 <==> gdal.GDT_CFloat32 == 10
            np.complex128 <==> gdal.GDT_CFloat64 == 11
            np.float <==> None
            np.int <==> None
            np.complex <==> None
        """
        nDataType_result = gdal_array.NumericTypeCodeToGDALTypeCode(nDataType)

        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_strImgPath = strImgPath

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0

        self.initNextWidowsSize()

        suffix = GetSuffix(self.m_strImgPath)
        if suffix == "bmp":
            driver = gdal.GetDriverByName("BMP")
        elif suffix == "jpg":
            driver = gdal.GetDriverByName("JPEG")
        elif suffix == "tif" or suffix == "tiff":
            driver = gdal.GetDriverByName("GTiff")
        elif suffix == "bt":
            driver = gdal.GetDriverByName("BT")
        elif suffix == "ecw":
            driver = gdal.GetDriverByName("ECW")
        elif suffix == "fits":
            driver = gdal.GetDriverByName("FITS")
        elif suffix == "gif":
            driver = gdal.GetDriverByName("GIF")
        elif suffix == "hdf":
            driver = gdal.GetDriverByName("HDF4")
        elif suffix == "hdr":
            driver = gdal.GetDriverByName("EHdr")
        elif suffix == "" or suffix == "img":
            driver = gdal.GetDriverByName("ENVI")
        else:
            print("GetDriverByName Error!")
            return

        # 保存图像到m_strImgPath路径，宽度为m_nSamples，高度为m_nLines，波段数为m_nBands，读入的数据为GDALDataType格式
        self.proDataset = driver.Create(self.m_strImgPath, nSamples, nLines, self.m_nBands,
                                        nDataType_result)

    # 打开文件，保存信息
    def Open(self, strImgPath):
        self.m_strImgPath = strImgPath

        self.proDataset = gdal.Open(self.m_strImgPath)
        if self.proDataset == None:
            print("file open error")
            return False
        self.m_nBands = self.proDataset.RasterCount
        self.m_nLines = self.proDataset.RasterYSize
        self.m_nSamples = self.proDataset.RasterXSize

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0
        self.currentHeight = self.m_nLines
        self.currentWidth = self.m_nSamples

        self.initNextWidowsSize()
        return True

    # 跳到下一个窗口
    def next(self, padding=10):
        self.currentPosX += self.currentWidth - padding
        self.currentPosY += 0

        if self.currentPosX + padding >= self.m_nSamples:
            self.currentPosX = 0
            self.currentPosY += self.currentHeight - padding
        if self.currentPosY + padding >= self.m_nLines:
            self.currentPosX = 0
            self.currentPosY = 0
            return False

        self.initNextWidowsSize()
        return True

    # 设置分块大小
    def setPartSize(self, setPartWidth=-1, setPartHeight=-1, memorySize=-1, typeSize=4, safetyFactor=6):
        # 设置了setPartWidth和setPartHeight参数
        if setPartWidth != -1 and setPartHeight != -1:
            self.partHeight = setPartHeight
            self.partWidth = setPartWidth
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        # 只设置了setPartWidth参数
        elif setPartWidth != -1 and setPartHeight == -1:
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = setPartWidth
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        # 只设置了memorySize参数，根据设置内存大小分块
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize != -1:
            if memorySize < 1000:
                self.partWidth = 0
                self.partHeight = 0
                return
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = int((memorySize * 1000.0 / (typeSize * ratio * safetyFactor)) ** 0.5)
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        # 无参数
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize == -1:
            self.partHeight = self.m_nLines
            self.partWidth = self.m_nSamples
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        else:
            print("setPartSize parameters error!")
            return

    # 获取分块数量
    def getPartionNum(self, padding=10):
        temp_currentPosX = self.currentPosX
        temp_currentPosY = self.currentPosY
        temp_currentHeight = self.currentHeight
        temp_currentWidth = self.currentWidth

        self.currentPosX = 0
        self.currentPosY = 0
        self.initNextWidowsSize()
        partionNum = 1
        maxPartionWidth = 0
        maxPartionHeight = 0

        while True:
            if maxPartionHeight < self.currentHeight:
                maxPartionHeight = self.currentHeight
            if maxPartionWidth < self.currentWidth:
                maxPartionWidth = self.currentWidth
            if not self.next(padding):
                break
            partionNum += 1
        self.currentPosX = temp_currentPosX
        self.currentPosY = temp_currentPosY
        self.currentHeight = temp_currentHeight
        self.currentWidth = temp_currentWidth
        return partionNum, maxPartionHeight, maxPartionWidth

    # 调整分块参数
    def setPartionParameters(self, currentPosXValue=-1, currentPosYValue=-1, currentWidthValue=-1,
                             currentHeightValue=-1):
        if currentHeightValue > 0 and currentPosYValue > -1 and currentPosXValue > -1 and currentWidthValue > 0:
            self.currentHeight = currentHeightValue
            self.currentWidth = currentWidthValue
            self.currentPosX = currentPosXValue
            self.currentPosY = currentPosYValue

    def setNormalizaion(self, setIsNormalization=True):
        self.isNormalization = setIsNormalization
        return

    # 设置头信息
    def setHeaderInformation(self, img):
        # 获取投影信息
        projectionRef = img.proDataset.GetProjectionRef()
        # 写入投影
        self.proDataset.SetProjection(projectionRef)

        # 获取仿射矩阵信息
        padfTransform = img.proDataset.GetGeoTransform()
        # 写入仿射变换参数
        self.proDataset.SetGeoTransform(padfTransform)

        # GCPCount = img.proDataset.GetGCPCount()
        # GCPProjection = img.proDataset.GetGCPProjection()
        # getGCPS = img.proDataset.GetGCPs()
        # self.proDataset.SetGCPs(GCPCount, getGCPS, GCPProjection)

    # 获取图像栅格数据，转化为dataType类型数组并返回
    def GetData(self, dataType, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1, data_arrange=0):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        # 没有设置cHeight、cWidth、cPosX、cPosY，即根据当前类内参数读取
        if cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:

            data = self.proDataset.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth,
                                               self.currentHeight)
        # 根据设定参数读取
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            data = self.proDataset.ReadAsArray(cPosX, cPosY, cWidth, cHeight)
        else:
            raise AttributeError("GetData parameters error!")

        if data.ndim == 3 and data_arrange == 0:
            data = data.transpose((1, 2, 0))
        elif data.ndim == 3 and data_arrange == 1:
            pass
        elif data.ndim == 2:
            pass
        else:
            raise AttributeError("data shape error!")

        data = data.astype(dataType)

        return data

    # 根据pData数据，并转化为Create时的类型对图像进行写入操作
    def WriteImgData(self, pData, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1, padding=10, data_arrange=0):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        # 直接写整幅图像
        if cHeight == self.m_nLines and cWidth == self.m_nSamples and cPosX == 0 and cPosY == 0:
            temp_currentPosX = self.currentPosX
            temp_currentPosY = self.currentPosY
            temp_currentWidth = self.currentWidth
            temp_currentHeight = self.currentHeight
            if data_arrange == 0 and pData.ndim == 3:
                temp_data = pData.transpose((2, 0, 1))
            else:
                temp_data = pData
        # 无参数图像写入
        elif cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:
            temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data = self.ProcessDataBeforeWrite(
                pData, self.currentHeight, self.currentWidth, self.currentPosX, self.currentPosY, padding, data_arrange)
        # 有参数图像写入
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data = self.ProcessDataBeforeWrite(
                pData, cHeight, cWidth, cPosX, cPosY, padding, data_arrange)
        else:
            raise AttributeError("WriteImgData parameters error!")

        temp_data = temp_data.astype(self.m_nDataType)
        self.proDataset.WriteRaster(int(temp_currentPosX), int(temp_currentPosY), int(temp_currentWidth),
                                    int(temp_currentHeight), temp_data.tobytes())
        self.proDataset.FlushCache()

        if not np.all(temp_data == 0):
            del temp_data
        return True

    # 对图像数据进行处理以正确写入
    def ProcessDataBeforeWrite(self, pData, cHeight, cWidth, cPosX, cPosY, padding, data_arrange):
        '''
            data_arrange == 0 <==> (height, width, band)
                         == 1 <==> (band, height, width)(gdal)
        '''
        temp_currentPosX = cPosX
        temp_currentPosY = cPosY
        temp_currentWidth = cWidth
        temp_currentHeight = cHeight
        if cPosX > 0:
            temp_currentPosX = cPosX + padding / 2
            temp_currentWidth = cWidth - padding / 2
        if cPosY > 0:
            temp_currentPosY = cPosY + padding / 2
            temp_currentHeight = cHeight - padding / 2
        if cPosX + cWidth < self.m_nSamples:
            temp_currentWidth = temp_currentWidth - padding / 2
        if cPosY + cHeight < self.m_nLines:
            temp_currentHeight = temp_currentHeight - padding / 2

        if data_arrange == 0 and pData.ndim == 3:
            pData = pData.transpose((2, 0, 1))

        if pData.ndim == 3:
            if cPosX == 0 and cPosY == 0:
                temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
            elif cPosX > 0 and cPosY == 0:
                temp_data = pData[:, :int(temp_currentHeight), int(padding / 2):int(temp_currentWidth + padding / 2)]
            elif cPosX > 0 and cPosY > 0:
                temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2), int(padding / 2):int(temp_currentWidth + padding / 2)]
            else:
                temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2), :int(temp_currentWidth)]
        elif pData.ndim == 2:
            if cPosX == 0 and cPosY == 0:
                temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth)]
            elif cPosX > 0 and cPosY == 0:
                temp_data = pData[:int(temp_currentHeight), int(padding / 2):int(temp_currentWidth + padding / 2)]
            elif cPosX > 0 and cPosY > 0:
                temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2), int(padding / 2):int(temp_currentWidth + padding / 2)]
            else:
                temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2), :int(temp_currentWidth)]
        else:
            raise AttributeError("pData.ndim error!")
        return temp_currentHeight, temp_currentWidth, temp_currentPosX, temp_currentPosY, temp_data

    # 初始化窗口大小，以及设置下一个窗口大小
    def initNextWidowsSize(self):
        acceptWidth = 0.5 * self.partWidth
        acceptHeight = 0.5 * self.partHeight

        self.currentWidth = self.partWidth
        self.currentHeight = self.partHeight

        if self.m_nSamples <= self.currentWidth:
            self.currentWidth = self.m_nSamples
        if self.m_nLines <= self.currentHeight:
            self.currentHeight = self.m_nLines

        if self.currentPosX + self.partWidth > self.m_nSamples or acceptWidth + self.currentPosX + self.partWidth > self.m_nSamples:
            self.currentWidth = self.m_nSamples - self.currentPosX
        if self.currentPosY + self.partHeight > self.m_nLines or acceptHeight + self.currentPosY + self.partHeight > self.m_nLines:
            self.currentHeight = self.m_nLines - self.currentPosY


# 将训练集和测试集按比例对每个类别进行划分(stochastic stratified sample)
# 对训练集设置(j的循环次数)次采样，每次采样将训练集打乱并取前若干个像素。
# 对测试集只取所有测试集像素的一张图
if __name__ == '__main__':
    # parameters & multiple band
    instrImgPath = r"C:\Users\Admin\Desktop\TDM1_DEM__30_S65W064_DEM_label.tif"
    xImgIn = CXImage()
    xImgIn.Open(instrImgPath)
    # padding = 0

    # data_preprocessing
    currentWidth = xImgIn.currentWidth
    currentHeight = xImgIn.currentHeight
    currentPosX = xImgIn.currentPosX
    currentPosY = xImgIn.currentPosY
    # xImgIn.next(padding)
    # 正常的代码处理，现在的输入数据是inImgData，大小是currentHeight,currentWidth
    inImgData = xImgIn.GetData(np.uint32, currentHeight, currentWidth, currentPosX, currentPosY, data_arrange=0)

    # # 采样前z-score normalization
    # inImgData = z_score_normalization(inImgData, dtype=np.float32, data_arrange=0)
    # 为采样后z-score标准化计算image每一波段的mean和std
    # mean, std = cal_mean_and_std_3d(inImgData, data_arrange=0)

    # 直接输出图像
    outImg = CXImage()
    strImgPath = r"C:\Users\Admin\Desktop\JAN_gt_1.tiff"
    outImg.Create(xImgIn.m_nBands, xImgIn.m_nLines, xImgIn.m_nSamples, np.uint32, strImgPath)
    outImg.WriteImgData(inImgData, currentHeight, currentWidth, currentPosX, currentPosY, padding=0, data_arrange=0)

