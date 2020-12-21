import numpy as np
import random

class PyROIComputation():

    # 样本点ID
    m_pPtsID = None
    # 存储各个类别ROI的点数
    m_pRoiPtsNum = None
    # 存储各类别各波段像素灰度值,按BIP的方式
    m_pPixelValue = None
    # 总的样本点数
    m_nSampleNum = 0
    # roi数据的波段数
    m_nImgBand = 0
    # roi类别数
    m_nRoiNum = 0
    # roi类名
    m_psRoiName = None

    def __init__(self):
        return

    # 计算第nRoiID类第nBandID波段样本的均值,nRoiID，nBandID均从零开始编号
    def GetRoiBandMean(self, nRoiID, nBandID):
        point_start = 0
        res = 0.0
        if np.all(self.m_pRoiPtsNum == 0) or np.all(self.m_pPixelValue == 0.0) or self.m_nRoiNum == 0:
            return -9999999
        for i in nRoiID:
            point_start += self.m_pRoiPtsNum[i]
        point_start = point_start * self.m_nImgBand
        point_num = self.m_pRoiPtsNum[nRoiID]
        for i in point_num:
            res += self.m_pPixelValue[point_start + i * self.m_nImgBand + nBandID]
        return res / point_num

    # 计算第nRoiID类样本的均值，nRoiID从零开始编号,pMeanData的大小m_nImgBand
    def GetRoiMean(self, pMeanData, nRoiID):
        if np.all(self.m_pRoiPtsNum == 0) or np.all(self.m_pPixelValue == 0.0) or self.m_nRoiNum == 0 or pMeanData is None:
            return 0
        point_start = 0
        for i in nRoiID:
            point_start += self.m_pRoiPtsNum[i]
        point_start = point_start * self.m_nImgBand
        point_num = self.m_pRoiPtsNum[nRoiID]
        for k in range(self.m_nImgBand):
            for i in point_num:
                pMeanData[k] += self.m_pPixelValue[point_start + i * self.m_nImgBand + k]
            pMeanData[k] /= point_num
        return

    # 计算第nRoiID类样本的方差，nRoiID从零开始编号,pMeanData的大小m_nImgBand
    def GetRoiCovariance(self, pCovData, pMeanData, nRoiID):
        if np.all(self.m_pRoiPtsNum == 0) or np.all(
                self.m_pPixelValue == 0.0) or self.m_nRoiNum == 0 or pMeanData is None:
            return False
        point_start = 0
        point_num = self.m_pRoiPtsNum[nRoiID]

        for i in nRoiID:
            point_start += self.m_pRoiPtsNum[i]
        point_start = point_start * self.m_nImgBand

        for i in point_num:
            iindex = point_start + i * self.m_nImgBand
            for k in range(self.m_nImgBand):
                for l in range(k, self.m_nImgBand):
                    pCovData[k * self.m_nImgBand + l] += (self.m_pPixelValue[iindex + k] - pMeanData[k]) * \
                                                    (self.m_pPixelValue[iindex + l] - pMeanData[l])
        for k in range(self.m_nImgBand):
            for l in range(k, self.m_nImgBand):
                pCovData[k * self.m_nImgBand + l] /= point_num
            pCovData[k * self.m_nImgBand + k] += 1e-6

        for k in range(self.m_nImgBand):
            for l in range(k):
                pCovData[k * self.m_nImgBand + l] = pCovData[l * self.m_nImgBand + k]
        return True

    # 计算各个类别样本的均值,pMeanData的大小应为H * W = m_nRoiNum * m_nImgBand
    def GetMeanStatics(self, pMeanData):
        if np.all(self.m_pRoiPtsNum == 0) or np.all(
                self.m_pPixelValue == 0.0) or self.m_nRoiNum == 0 or pMeanData is None:
            return False
        for i in range(self.m_nRoiNum):
            if not self.GetRoiMean(pMeanData[i*self.m_nImgBand], i):
                return False
        return True

    # 将各类ROI样本点的顺序随机打乱
    def RandDisturbROISamples(self):
        i = 0
        idroi = 0
        j = 0
        idpoint = 0
        while i < self.m_nRoiNum:
            while j < self.m_pRoiPtsNum[i]:
                index = random.randint(1, 1000) % (self.m_pRoiPtsNum[i] - j) + j
                indexpoint = index * self.m_nImgBand
                if index != j:
                    for k in range(self.m_nImgBand):
                        temp = self.m_pPixelValue[idroi + idpoint + k]
                        self.m_pPixelValue[idroi + idpoint + k] = self.m_pPixelValue[idroi + indexpoint + k]
                        self.m_pPixelValue[idroi + indexpoint + k] = temp
                j += 1
                idpoint += self.m_nImgBand
            i += 1
            idroi += self.m_pRoiPtsNum[i]*self.m_nImgBand
        return

    # 根据样本的坐标和输入的影像更新样本点的值
    # void UpdatePtsVal(const double* pImage, int size, int band);

    #给定最大最小值对roi进行归一化到[-1,1], fmax,fmin为拉伸前影像的最大最小值， gmax，gmin为拉伸后的最大最小值。
    def NormPixelValue(self, fmax, fmin, gmax, gmin):
        if fmax < fmin or gmax < gmin:
            return
        datacount = self.m_nImgBand * self.m_nSampleNum
        fspan_t = fmax - fmin
        gspan_t = gmax - gmin

        if fmin == fmax:
            for i in range(datacount):
                self.m_pPixelValue[i] = gmin
        else:
            for i in range(datacount):
                self.m_pPixelValue[i] = (self.m_pPixelValue[i] - fmin) * gspan_t / fspan_t - 1
        return

    # 获取类成员变量函数
    def GetImgBandNum(self):
        return self.m_nImgBand

    def GetRoiNum(self):
        return self.m_nRoiNum

    def GetRoiName(self):
        return self.m_psRoiName

    def GetSampleNum(self):
        return self.m_nSampleNum

    def GetRoiPtsNum(self):
        return self.m_pRoiPtsNum

    def GetPixelValue(self):
        return self.m_pPixelValue

    def GetPtsID(self):
        return self.m_pPtsID

    # 打开ENVI导出的ROI文件，并求取灰度均值信息,要求必须输出ROI中点的ID
    def Open(self, strRoiPath, gtRoot="", imgRoot=""):
        return

    def checkROI(self, strRoiPath, classNum, imgHeight, imgWidth):


        return

    # init()函数与Open的功能一样，都用于初始化ROI的成员变量
    # nRoiNum, nImgBand, nSampleNum分别表示ROI的类别数、数据的波段数，总的样本点数
    # pRoiPtsNum, pPivelValue分别存储各个类别ROI中的样本数，按照类别顺序存储的所有样本点特征向量
    def init(self, nRoiNum, nImgBand, nSampleNum, pRoiPtsNum, pPixelValue, psRoiName=""):
        return

    # 合并样本类
    def MergeROI(self, roi):

        if roi.GetRoiNum() == 0:
            return True
        if self.m_nRoiNum == 0:
            self.init(roi.GetRoiNum(), roi.GetImgBandNum(), roi.GetSampleNum(), roi.GetRoiPtsNum(), roi.GetPixelValue(),
                                                                                                        roi.GetRoiName())
            return True
        if roi.GetImgBandNum() != self.m_nImgBand:
            print("the band number of image roi merged is not consistent!")
            return False

        roinum1 = roi.GetRoiNum()
        roinum2 = self.m_nRoiNum
        roiname1 = roi.GetRoiName()
        roiname2 = self.m_psRoiName
        bexist = False

        for i in roinum1:
            for j in range(roinum2):
                if roiname1[i] == roiname2[j]:
                    bexist = True
            if not bexist:
                self.m_nRoiNum += 1
            bexist = False
        # 更新各个roi的名称
        for i in range(roinum2):
            self.m_psRoiName[i] = roiname2[i]
            del roiname2[i]
        del roiname2
        roiname2 = None
        count = roinum2
        for i in roinum1:
            for j in range(roinum2):
                if roiname1[i] == self.m_psRoiName[j]:
                    bexist = True
            if not bexist:
                self.m_psRoiName[count] = roiname1[i]
                count += 1
            bexist = False

        samplenum1 = roi.GetSampleNum()     # 更新各个roi中的样本数,总样本数，对应的特征数
        self.m_nSampleNum += samplenum1     # 更新总样本数

        roiptsnum1 = roi.GetRoiPtsNum()
        roiptsnum2 = self.m_pRoiPtsNum
        pixelvalue1 = roi.GetPixelValue()
        pixelvalue2 = self.m_pPixelValue

        self.m_pRoiPtsNum = np.zeros(self.m_nRoiNum, dtype=int)
        self.m_pPixelValue = np.zeros(self.m_nSampleNum*self.m_nImgBand, dtype=float)

        jid = 0
        iid2 = 0
        iid1 = 0
        for j in range(self.m_nRoiNum):
            if j < roinum2:
                self.m_pRoiPtsNum[j] = roiptsnum2[j]
                kid = roiptsnum2[j] * self.m_nImgBand
                for i in kid:
                    self.m_pPixelValue[jid] = pixelvalue2[iid2]
                    jid += 1
                    iid2 += 1
            for i in roinum1:
                if roiname1[i] != self.m_psRoiName[j]:
                    continue
                self.m_pRoiPtsNum[j] += roiptsnum1[i]
                kid = roiptsnum1[i] * self.m_nImgBand
                for k in kid:
                    self.m_pPixelValue[jid] = pixelvalue1[iid1]
                    jid += 1
                    iid1 += 1
                break

        del roiptsnum2
        del pixelvalue2

        return True

    # 随机选择样本用作训练
    def randSelectedSamples(self, ratio = 0.05, minSampleNum = 100):
        classRatio = 0.0
        maxSampleNum = 0
        self.m_nSampleNum = 0
        m_pRoiPtsNumTemp = np.zeros(self.m_nRoiNum, dtype=int)

        for i in range(self.m_nRoiNum):
            classRatio = minSampleNum * 1.0 / self.m_pRoiPtsNum[i]
            if classRatio < ratio:
                classRatio = ratio
            m_pRoiPtsNumTemp[i] = int(self.m_pRoiPtsNum[i] * classRatio)
            self.m_nSampleNum += m_pRoiPtsNumTemp[i]
            if self.m_pRoiPtsNum[i] > maxSampleNum:
                maxSampleNum = self.m_pRoiPtsNum[i]
        randNum = np.zeros(maxSampleNum, dtype=int)

        for i in maxSampleNum:
            randNum[i] = i
        self.m_pPtsIDTemp = np.zeros(self.m_nSampleNum*2, dtype=int)
        IDIndex = 0
        beforeSampleNums = 0

        for i in range(self.m_nRoiNum):
            print("还没写完~")
        return

    def UpdatePtsVal(self, pImage, height, width, band):
        return

    def SpecialUpdatePtsVal(self, img, istexture=False, isNormalization=True, nBandRed=2, nBandGreen=1, nBandInfrared=3):
        return
