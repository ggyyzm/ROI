from XImage import CXImage
import numpy as np


def cut_image(inImgPath, outImgPath, cPosX, cPosY, cHeight, cWidth):
    inImg = CXImage()
    inImg.Open(inImgPath)
    inImgData = inImg.GetData(np.uint32, inImg.currentHeight, inImg.currentWidth, inImg.currentPosX, inImg.currentPosY, data_arrange=0)

    outImg = CXImage()
    outImg.Create(inImg.m_nBands, cHeight, cWidth, np.uint32, outImgPath)
    outImg.WriteImgData(inImgData, cHeight, cWidth, cPosX, cPosY, padding=0, data_arrange=0)

    return


if __name__ == '__main__':
    instrImgPath = r"F:\Antarctic_cfl\global_dem_90_coast\TDM1_DEM__30_S77W032_DEM.tif"       # 图像路径
    instrGtPath = r"F:\Antarctic_cfl\coastline_label_1207\TDM1_DEM__30_S77W032_DEM_label.tif"     # groundtruth路径

    outstrImgPath = r"F:\Antarctic_cfl\global_dem_90_coast_1221\TDM1_DEM__30_S77W032_DEM_0.tif"     # 输出图像路径
    outstrGtPath = r"F:\Antarctic_cfl\coastline_label_1221\TDM1_DEM__30_S77W032_DEM_label_0.tif"    # 输出groundtruth路径

    PosX = 200      # 左上角顶点X坐标
    PosY = 200      # 左上角顶点Y坐标
    cut_Height = 200    # 裁剪高度
    cut_Width = 200  # 裁剪宽度

    cut_image(instrImgPath, outstrImgPath, PosX, PosY, cut_Height, cut_Width)
    cut_image(instrGtPath, outstrGtPath, PosX, PosY, cut_Height, cut_Width)

