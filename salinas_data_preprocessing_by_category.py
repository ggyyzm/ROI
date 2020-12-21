import numpy as np
from data_preporcess import indexToAssignment, sampling_by_category, generate_array, z_score_normalization
from XImage import CXImage

# 将训练集和测试集按比例对每个类别进行划分(stochastic stratified sample)
# 将训练集中每个类别的点单独分出来(保存在train_assign_by_category中)
# 对训练集设置(j的循环次数)次采样，每次采样将训练集每个类别的像素分别打乱并取前若干个像素。
# 对测试集只取所有测试集像素的一张图
if __name__ == '__main__':
    NUM_OF_EACH_CLASS = 200

    # parameters & multiple band
    instrImgPath = r"C:\Users\Admin\Desktop\JAN_train.tiff"
    xImgIn = CXImage()
    xImgIn.Open(instrImgPath)
    # padding = 0
    num_classes = 16

    # groundtruth
    xGroundTruthImgPath = r"C:\Users\Admin\Desktop\JAN_train_gt.tiff"
    xGroundTruthImg = CXImage()
    xGroundTruthImg.Open(xGroundTruthImgPath)
    groundTruthData = xGroundTruthImg.GetData(np.uint32, data_arrange=0)

    # data_preprocessing
    currentWidth = xImgIn.currentWidth
    currentHeight = xImgIn.currentHeight
    currentPosX = xImgIn.currentPosX
    currentPosY = xImgIn.currentPosY
    # xImgIn.next(padding)
    # 正常的代码处理，现在的输入数据是inImgData，大小是currentHeight,currentWidth
    inImgData = xImgIn.GetData(np.float32, currentHeight, currentWidth, currentPosX, currentPosY, data_arrange=0)
    # z-score normalization
    inImgData = z_score_normalization(inImgData, dtype=np.float32, data_arrange=0)

    np.random.seed(200)
    train_indices, test_indices = sampling_by_category(NUM_OF_EACH_CLASS, groundTruthData)
    train_assign = indexToAssignment(train_indices, groundTruthData.shape[0], groundTruthData.shape[1])
    test_assign = indexToAssignment(test_indices, groundTruthData.shape[0], groundTruthData.shape[1])

    # train_assign_by_category = divide_by_category(train_assign, groundTruthData)

    # train
    for j in range(1):
        np.random.seed(j)
        # temp_train_assign = []
        # for i in range(1, num_classes+1):
        #     np.random.shuffle(train_assign_by_category[i])
        #     temp_train_assign += list(train_assign_by_category[i].values())[:200]

        # 生成训练集label的mask
        train_label_random_array = generate_array(train_assign, groundTruthData.shape[0],
                                                  groundTruthData.shape[1])  # temp_train_assign
        # 生成训练集image的mask
        train_image_random_array = np.array([train_label_random_array, train_label_random_array])
        temp_train_label_random_array = np.array([train_label_random_array])
        # 针对多波段
        for k in range(202):
            train_image_random_array = np.concatenate((train_image_random_array, temp_train_label_random_array), axis=0)
        train_image_random_array = train_image_random_array.transpose((1, 2, 0))

        # 根据训练集image的mask对imageData进行处理，将忽略的pixel置0
        temp_image_data = inImgData * train_image_random_array
        temp_image_data = temp_image_data.astype(np.float32)
        # 根据训练集label的mask对groundTruthData进行处理，将忽略的pixel置0
        temp_seg_data = groundTruthData * train_label_random_array
        temp_seg_data = temp_seg_data.astype(np.uint32)

        outImg = CXImage()
        strImgPath = r"C:\Users\Admin\Desktop\20200114_204d_200pclassTrainingset\P" + str(j) + ".tiff"
        outImg.Create(xImgIn.m_nBands, xImgIn.m_nLines, xImgIn.m_nSamples, np.float32, strImgPath)
        outImg.WriteImgData(temp_image_data, currentHeight, currentWidth, currentPosX, currentPosY, padding=0,
                            data_arrange=0)
        del outImg

        outImg_gt = CXImage()
        strImgPath_gt = r"C:\Users\Admin\Desktop\20200114_204d_200pclassTrainingsetgt\P" + str(j) + ".tiff"
        outImg_gt.Create(1, xImgIn.m_nLines, xImgIn.m_nSamples, np.uint32, strImgPath_gt)
        outImg_gt.WriteImgData(temp_seg_data, currentHeight, currentWidth, currentPosX, currentPosY, padding=0,
                               data_arrange=0)
        # del inImgData
        # outImg.setHeaderInformation(xImgIn)
        del outImg_gt

    # test
    test_label_random_array = generate_array(test_assign, groundTruthData.shape[0], groundTruthData.shape[1])
    test_image_random_array = np.array([test_label_random_array, test_label_random_array])
    temp_test_label_random_array = np.array([test_label_random_array])
    for k in range(202):
        test_image_random_array = np.concatenate((test_image_random_array, temp_test_label_random_array), axis=0)
    test_image_random_array = test_image_random_array.transpose((1, 2, 0))

    # 根据测试集image的mask对imageData进行处理，将忽略的pixel置0
    temp_image_data = inImgData * test_image_random_array
    temp_image_data = temp_image_data.astype(np.float32)
    # 根据测试集label的mask对groundTruthData进行处理，将忽略的pixel置0
    temp_seg_data = groundTruthData * test_label_random_array
    temp_seg_data = temp_seg_data.astype(np.uint32)

    outImg = CXImage()
    strImgPath = r"C:\Users\Admin\Desktop\20200114_204d_200pclassTrainingset\P1000.tiff"
    outImg.Create(xImgIn.m_nBands, xImgIn.m_nLines, xImgIn.m_nSamples, np.float32, strImgPath)
    outImg.WriteImgData(temp_image_data, currentHeight, currentWidth, currentPosX, currentPosY, padding=0,
                        data_arrange=0)
    del outImg

    outImg_gt = CXImage()
    strImgPath_gt = r"C:\Users\Admin\Desktop\20200114_204d_200pclassTrainingsetgt\P1000.tiff"
    outImg_gt.Create(1, xImgIn.m_nLines, xImgIn.m_nSamples, np.uint32, strImgPath_gt)
    outImg_gt.WriteImgData(temp_seg_data, currentHeight, currentWidth, currentPosX, currentPosY, padding=0,
                           data_arrange=0)
    # del inImgData
    # outImg.setHeaderInformation(xImgIn)

    del outImg_gt
