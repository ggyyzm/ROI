from osgeo import gdal
import numpy as np
from osgeo import gdal_array
import os
import cmath
import copy

def GetSuffix(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i+1].isdigit() or len(filepath) - i > 5:
        return str
    else:
        str = filepath[i+1:len(filepath)]
        return str

def GetFilePath(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i+1].isdigit():
        return filepath
    else:
        str = filepath[0:i]
        return str

class CXImage():

    m_nBands = 0    # 波段数
    m_nLines = 0    # height
    m_nSamples = 0  # width
    m_nClasses = 0

    m_nDataType = np.uint8
    m_strImgPath = ""

    currentHeight = 0
    currentWidth = 0
    partWidth = 0
    partHeight = 0
    isNormalization = True

    proDataset = None
    currentPosX = 0
    currentPosY = 0

    minBandValue = None
    maxBandValue = None

    def __init__(self, nBands=0, nLines=0, nSamples=0, nDataType=np.uint8, nClass=0, strImgPath=None):
        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_nClasses = nClass
        self.m_strImgPath = strImgPath

        self.minBandValue = None
        self.maxBandValue = None
        self.isNormalization = False
        self.proDataset = None

    def Create(self, nBands, nLines, nSamples, nDataType, strImgPath, nClass, padding=10, temp_image=None):
        if temp_image == None:
            temp_image = CXImage()
            temp_image.Open(self.m_strImgPath)
        # self.m_nBands = nBands
        # self.m_nLines = nLines
        # self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_nClasses = nClass
        self.m_strImgPath = strImgPath

        # self.partWidth = self.m_nSamples
        # self.partHeight = self.m_nLines
        # self.currentPosX = 0
        # self.currentPosY = 0
        # self.currentHeight = self.m_nLines
        # self.currentWidth = self.m_nSamples
        #
        # self.initNextWidowsSize()
        # self.setPartSize(setPartWidth=setPartWidth, setPartHeight=setPartHeight, memorySize=memorySize)

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

        if self.currentWidth != self.m_nSamples and self.currentHeight != self.m_nLines:
            temp_currentWidth = self.currentWidth
            temp_currentHeight = self.currentHeight
            if self.currentPosX > 0:
                temp_currentWidth = self.currentWidth - padding / 2
            if self.currentPosY > 0:
                temp_currentHeight = self.currentHeight - padding / 2
            if self.currentPosX + self.currentWidth < self.m_nSamples:
                temp_currentWidth = temp_currentWidth - padding / 2
            if self.currentPosY + self.currentHeight < self.m_nLines:
                temp_currentHeight = temp_currentHeight - padding / 2
            self.proDataset = driver.Create(self.m_strImgPath, int(temp_currentWidth), int(temp_currentHeight), nBands,
                                            self.m_nDataType)
        else:
            # 保存图像到m_strImgPath路径，宽度为nSamples，高度为nLines，波段数为nBands，读入的数据转换为self.m_nDataType格式？
            self.proDataset = driver.Create(self.m_strImgPath, nSamples, nLines, nBands, self.m_nDataType)

        self.setHeaderInformation(img=temp_image)
        self.WriteImgData(pData=temp_image.GetData(dataType=self.m_nDataType, cWidth=self.currentWidth, cHeight=self.currentHeight, cPosX=self.currentPosX, cPosY=self.currentPosY)
                          , cWidth=self.currentWidth, cHeight=self.currentHeight, cPosX=self.currentPosX, cPosY=self.currentPosY, padding=padding)

        if self.m_nBands == 1 and self.m_nClasses > 0:
            self.setClassificationColor()

        minBandValue = None
        maxBandValue = None

        return
        # if self.proDataset != None:
        #     del self.proDataset
        #
        # # 注册所有已知的驱动
        # gdal.AllRegister()
        #
        # self.proDataset = gdal.Open(self.m_strImgPath)
        # if self.proDataset == None:
        #     print("file open error")
        #     return

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

    def setPartSize(self, setPartWidth=-1, setPartHeight=-1, memorySize=-1, typeSize=4, safetyFactor=6):
        if setPartWidth != -1 and setPartHeight != -1:
            self.partHeight = setPartHeight
            self.partWidth = setPartWidth
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        elif setPartWidth != -1 and setPartHeight == -1:
            ratio = float(self.m_nLines)/self.m_nSamples
            self.partWidth = setPartWidth
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize != -1:
            if memorySize < 1000:
                self.partWidth = 0
                self.partHeight = 0
                return
            ratio = float(self.m_nLines)/self.m_nSamples
            self.partWidth = int((memorySize * 1000.0/(typeSize * ratio * safetyFactor))**0.5)
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize == -1:
            self.partHeight = self.m_nLines
            self.partWidth = self.m_nSamples
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        else:
            print("setPartSize parameters error!")
            return

    def setClassNo(self, classNo):
        self.m_nClasses = classNo
        if self.m_nBands == 1 and self.m_nClasses > 0:
            self.setClassificationColor()
        return

    def setClassificationColor(self, trainFile=""):
        isUsedDefault = False
        if trainFile == "":
            isUsedDefault = True
        else:
            Is = open(trainFile, "r")
            if Is == None:
                print("Open ROI file Error!")
                isUsedDefault = True
            pIn = ""
        if not isUsedDefault:
            pIn = Is.readline()
            pIn = Is.readline()
            self.m_nClasses = pIn.split()[4]
            pIn = Is.readline()

        usedColor = np.zeros(3*(self.m_nClasses+1), dtype=int)
        pColor = [0,0,0,  0,0,255,  46,139,87,  0,255,0,  216,191,216,  255,0,0,  255,255,255,
        255,255,0, 0,255,255,  255,0,255,  48,48,48,  128,0,0,  0,128,0,	0,0,128,  128,128,0,  0,128,128,
        128,0,128,  255,128,0,  128,255,0,  255,0,128]

        colorTable = gdal.ColorTable()
        pBand = self.proDataset.GetRasterBand(1)

        for i in range(self.m_nClasses+1):
            if isUsedDefault or i==0:
                usedColor[3 * i] = pColor[(3*i) % 60]
                usedColor[3 * i + 1] = pColor[(3 * i + 1) % 60]
                usedColor[3 * i + 2] = pColor[(3 * i + 2) % 60]
            else:
                pIn = Is.readline()     # 读空行
                pIn = Is.readline()     # ; ROI name: Random Sample (salinas_gt_byte / Class #1)
                pIn = Is.readline()     # ; ROI rgb value: {255, 0, 0}
                strTemp = pIn.split()[4]    # {255,
                str = strTemp[1:len(strTemp)-1]     # 读255
                usedColor[3 * i] = int(str)

                strTemp = pIn.split()[5]    # 0,
                str = strTemp[0:1]
                usedColor[3 * i + 1] = int(str)

                strTemp = pIn.split()[6]    # 0}
                str = strTemp[0:1]
                usedColor[3 * i + 2] = int(str)

                pIn = Is.readline()     # ; ROI npts: 201
            colorEntry = gdal.ColorEntry()
            colorEntry.c1 = usedColor[3*i]
            colorEntry.c2 = usedColor[3 * i + 1]
            colorEntry.c3 = usedColor[3 * i + 2]
            colorEntry.c4 = 0
            colorTable.SetColorEntry(i, colorEntry)

        return pBand.SetColorTable(colorTable)

    def getPartionNum(self, maxPartionWidth, maxPartionHeight, padding=10):
        temp_currentPosX = self.currentPosX
        temp_currentPosY = self.currentPosY
        temp_currentHeight = self.currentHeight
        temp_currentWidth = self.currentWidth

        self.currentPosX = 0
        self.currentPosY = 0
        self.initNextWidowsSize()
        partionNum = 1

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
        return partionNum

    def setPartionParameters(self, currentPosXValue=-1, currentPosYValue=-1, currentWidthValue=-1, currentHeightValue=-1):
        if currentHeightValue > 0 and currentPosYValue > -1 and currentPosXValue > -1 and currentWidthValue > 0:
            self.currentHeight = currentHeightValue
            self.currentWidth = currentWidthValue
            self.currentPosX = currentPosXValue
            self.currentPosY = currentPosYValue

    def initMinMaxBandValue(self, is2PercentScale=False, percentValue=0.02):
        return

    def uninitMinMaxBandValue(self):
        return

    def setNormalizaion(self, setIsNormalization=True):
        self.isNormalization = setIsNormalization
        return

    def setHeaderInformation(self, img):
        projectionRef = img.proDataset.GetProjectionRef()
        self.proDataset.SetProjection(projectionRef)

        padfTransform = img.proDataset.GetGeoTransform()
        self.proDataset.SetGeoTransform(padfTransform)

        # GCPCount = img.proDataset.GetGCPCount()
        # GCPProjection = img.proDataset.GetGCPProjection()
        # getGCPS = img.proDataset.GetGCPs()
        # self.proDataset.SetGCPs(GCPCount, getGCPS, GCPProjection)

    def GetData(self, dataType, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1, isParallel=False):
        if dataType == 0:
            dataType_result = None
        elif dataType == 1:
            dataType_result = np.uint8
        elif dataType == 2:
            dataType_result = np.uint16
        elif dataType == 3:
            dataType_result = np.int16
        elif dataType == 4:
            dataType_result = np.uint32
        elif dataType == 5:
            dataType_result = np.int32
        elif dataType == 6:
            dataType_result = np.float32
        elif dataType == 7:
            dataType_result = np.float64
        elif dataType == 8:
            dataType_result = np.complex64
        elif dataType == 9:
            dataType_result = np.complex64
        elif dataType == 10:
            dataType_result = np.complex64
        elif dataType == 11:
            dataType_result = np.complex128
        else:
            dataType_result = dataType
        # 没有设置cHeight、cWidth、cPosX、cPosY表示全读
        if cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:
            data = self.proDataset.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth, self.currentHeight)
            # bandMap = np.zeros(self.m_nBands, dtype=int)
            # for i in range(1, self.m_nBands+1):
            #     # bandMap[i - 1] = i
            #     band = self.proDataset.GetRasterBand(i)
            #     if i == 1:
            #         data = band.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth, self.currentHeight)
            #         data = np.array([data])
            #     else:
            #         data = np.concatenate((data, np.array([band.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth, self.currentHeight)])), axis=0)
            # # del bandMap
            data_result = data.astype(dataType_result)
            # return data_result
            return data_result
        # 分块读取，返回以点cPosX、cPosY为左上角顶点，cWidth、cHeight为宽高，m_nBands为波段数，dataType为数据类型的三维数组数据
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            # bandMap = np.zeros(self.m_nBands, dtype=int)
            # for i in range(1, self.m_nBands + 1):
            #     bandMap[i - 1] = i

            proDatasetTemp = None
            if isParallel:
                proDatasetTemp = gdal.Open(self.m_strImgPath)
                if proDatasetTemp == None:
                    isParallel = False
                else:
                    data = proDatasetTemp.ReadAsArray(cPosX, cPosY, cWidth, cHeight)
            if proDatasetTemp != None:
                del proDatasetTemp
            if not isParallel:
                data = self.proDataset.ReadAsArray(cPosX, cPosY, cWidth, cHeight)
            data_result = data.astype(dataType_result)
            # del bandMap
            return data_result
        else:
            print("GetData parameters error!")
            return

    def NormlizedData(self, data, setMin=0, setMax=1):
        return

    def WriteImgData(self, pData, padding=10, cHeight=-1, cWidth=-1, cPosX=0, cPosY=0, isParallel=False):
        # 非分块输出
        if cHeight == (self.m_nLines or -1) and cWidth == (self.m_nSamples or -1) and cPosX == 0 and cPosY == 0:
            # bandMap = np.zeros(self.m_nBands, dtype=int)
            # for i in range(1, self.m_nBands+1):
            #     bandMap[i-1] = i

            # temp_currentPosX = self.currentPosX
            # temp_currentPosY = self.currentPosY
            # temp_currentWidth = self.currentWidth
            # temp_currentHeight = self.currentHeight
            # if self.currentPosX > 0:
            #     temp_currentPosX = self.currentPosX + padding/2
            #     temp_currentWidth = self.currentWidth - padding/2
            # if self.currentPosY > 0:
            #     temp_currentPosY = self.currentPosY + padding/2
            #     temp_currentHeight = self.currentHeight + padding/2
            #
            # if self.currentPosY > 0 or self.currentPosX > 0:
            #     temp_size = temp_currentHeight * temp_currentWidth
            #     temp_data = np.zeros(temp_size*self.m_nBands)
            #     temp_data = np.array([temp_data])
            #     for k in range(self.m_nBands):
            #         for i in range(temp_currentHeight):
            #             for j in range(temp_currentWidth):
            #                 temp_data[j + i * temp_currentWidth + k * temp_size] = pData[
            #                     (j + temp_currentPosX - self.currentPosX) + (
            #                             i + temp_currentPosY - self.currentPosY) * self.currentWidth + k * self.currentWidth * self.currentHeight]
            # else:
            temp_data = pData
            for i in range(1, self.m_nBands+1):
                self.proDataset.GetRasterBand(i).WriteArray(temp_data[i-1])
            self.proDataset.FlushCache()
            # del bandMap
            if self.currentPosY > 0 or self.currentPosX > 0:
                if temp_data != None:
                    del temp_data
            return True
        # 分块输出
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            # bandMap = np.zeros(self.m_nBands, dtype=int)
            # for i in range(1, self.m_nBands):
            #     bandMap[i-1] = i
            temp_currentPosX = self.currentPosX
            temp_currentPosY = self.currentPosY
            temp_currentWidth = self.currentWidth
            temp_currentHeight = self.currentHeight
            if cPosX > 0:
                temp_currentPosX = cPosX + padding / 2
                temp_currentWidth = cWidth - padding / 2
            if cPosY > 0:
                temp_currentPosY = cPosY + padding / 2
                temp_currentHeight = cHeight - padding / 2
            if cPosX + cWidth < self.m_nSamples:
                temp_currentWidth = temp_currentWidth - padding / 2
            if cPosY+cHeight < self.m_nLines:
                temp_currentHeight = temp_currentHeight - padding / 2
            # temp_size = temp_currentHeight * temp_currentWidth
            temp_data = np.random.randint(1, size=(self.m_nBands, int(temp_currentHeight), int(temp_currentWidth)))
            temp_data = temp_data.astype(pData.dtype)
            # temp_data = np.zeros(temp_size * self.m_nBands)
            # temp_data = np.array([temp_data])
            for k in range(self.m_nBands):
                for i in range(int(temp_currentHeight)):
                    for j in range(int(temp_currentWidth)):
                        temp_data[k][i][j] = pData[k][i + int(temp_currentPosY) - cPosY][j + int(temp_currentPosX) - cPosX]
            proDatasetTemp = None
            if isParallel:
                proDatasetTemp = gdal.Open(self.m_strImgPath)
                if proDatasetTemp == None:
                    isParallel = False
                else:
                    for i in range(1, self.m_nBands + 1):
                        proDatasetTemp.GetRasterBand(i).WriteArray(temp_data[i - 1])
            if proDatasetTemp != None:
                del proDatasetTemp
            if not isParallel:
                for i in range(1, self.m_nBands + 1):
                    self.proDataset.GetRasterBand(i).WriteArray(temp_data[i - 1])

            # del bandMap
            if np.all(temp_data == 0):
                del temp_data
            return True
        else:
            print("WriteImgData parameters error!")
            return False

    def Multi_Create(self, nBands, nDataType, strImgPath, nClass, maxPartionWidth, maxPartionHeight, padding=10):
        temp_image = CXImage()
        temp_image.Open(self.m_strImgPath)
        for i in range(self.getPartionNum(maxPartionWidth=maxPartionWidth, maxPartionHeight=maxPartionHeight, padding=padding)):
            self.Create(nBands=nBands, nLines=self.currentHeight, nSamples=self.currentWidth, nDataType=nDataType, strImgPath=strImgPath+r"\P"+str(i)+".tiff", nClass=nClass, temp_image=temp_image, padding=padding)
            self.next(padding)
        return

    def initNextWidowsSize(self):
        acceptWidth = 0.5*self.partWidth
        acceptHeight = 0.5*self.partHeight

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

if __name__ == '__main__':
    # 分块输出
    image = CXImage()
    image.Open(r"C:\Users\Admin\Desktop\salinas_byte")
    image.setPartSize(setPartWidth=20, setPartHeight=20)
    image.Multi_Create(nBands=3, nDataType=gdal.GDT_Byte, strImgPath=r"C:\Users\Admin\Desktop\20190929", nClass=0, maxPartionWidth=30, maxPartionHeight=30, padding=4)
    del image
    # image.Create(nBands=3, nLines=217, nSamples=512, nDataType=gdal.GDT_Byte, strImgPath=r"C:\Users\Admin\Desktop\S4.tiff", nClass=0, temp_image=temp_image)    # 1605 , 2602

    # 非分块输出
    # image = CXImage()
    # image.Open(r"C:\Users\Admin\Desktop\new\subset_result")
    # # image.Create(nBands=9, nLines=1605, nSamples=2602, nDataType=gdal.GDT_Byte, strImgPath=r"C:\Users\Admin\Desktop\3.tiff", nClass=0)
    # image.GetData(gdal.GDT_Byte)
    # del image







