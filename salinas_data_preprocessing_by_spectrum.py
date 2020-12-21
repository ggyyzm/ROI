import numpy as np
from data_preporcess import indexToAssignment, sampling_by_category, generate_array, z_score_normalization
from XImage import CXImage

# 将训练集和测试集按比例对每个类别进行划分(sampling) 或 将训练集按每类别固定num个数量进行划分(sampling_by_category)
# 对训练集设置(j的循环次数)次采样，每次采样将训练集的波段打乱并取前3个波段。
# 对测试集取相同波段
if __name__ == '__main__':
    NUM_OF_EACH_CLASS = 200

    # parameters & multiple band
    instrImgPath = r"C:\Users\Admin\Desktop\salinas.envi"
    xImgIn = CXImage()
    xImgIn.Open(instrImgPath)
    padding = 0
    num_classes = 16

    # groundtruth
    xGroundTruthImgPath = r"C:\Users\Admin\Desktop\salinas_gt.envi"
    xGroundTruthImg = CXImage()
    xGroundTruthImg.Open(xGroundTruthImgPath)
    groundTruthData = xGroundTruthImg.GetData(np.uint32, data_arrange=0)

    # data_preprocessing
    currentWidth = xImgIn.currentWidth
    currentHeight = xImgIn.currentHeight
    currentPosX = xImgIn.currentPosX
    currentPosY = xImgIn.currentPosY
    xImgIn.next(padding)
    # 正常的代码处理，现在的输入数据是inImgData，大小是currentHeight,currentWidth
    inImgData = xImgIn.GetData(np.float32, currentHeight, currentWidth, currentPosX, currentPosY, data_arrange=0)
    # z-score normalization
    inImgData = z_score_normalization(inImgData, dtype=np.float32, data_arrange=0)

    np.random.seed(200)
    train_indices, test_indices = sampling_by_category(NUM_OF_EACH_CLASS, groundTruthData)
    train_assign = indexToAssignment(train_indices, groundTruthData.shape[0], groundTruthData.shape[1])
    test_assign = indexToAssignment(test_indices, groundTruthData.shape[0], groundTruthData.shape[1])

    # train
    for j in range(128):
        np.random.seed(j)

        # 创建光谱随机选择数组
        spectrum = np.arange(204, dtype=np.uint8)
        np.random.shuffle(spectrum)
        spectrum_selection = spectrum[:3]

        # 对inImgData进行波段选择
        selected_inImgData = np.array((inImgData[:, :, spectrum_selection[0]], inImgData[:, :, spectrum_selection[1]],
                                       inImgData[:, :, spectrum_selection[2]]))
        selected_inImgData = selected_inImgData.transpose((1, 2, 0))

        # 生成训练集label的mask
        train_label_random_array = generate_array(train_assign, groundTruthData.shape[0], groundTruthData.shape[1])
        # 生成训练集image的mask
        train_image_random_array = np.array([train_label_random_array, train_label_random_array])
        temp_train_label_random_array = np.array([train_label_random_array])
        # 针对多波段提高mask的第三维度
        for k in range(1):
            train_image_random_array = np.concatenate((train_image_random_array, temp_train_label_random_array), axis=0)
        train_image_random_array = train_image_random_array.transpose((1, 2, 0))

        # 根据训练集image的mask对imageData进行处理，将忽略的pixel置0
        temp_image_data = selected_inImgData * train_image_random_array
        temp_image_data = temp_image_data.astype(np.float32)
        # 根据训练集label的mask对groundTruthData进行处理，将忽略的pixel置0
        temp_seg_data = groundTruthData * train_label_random_array
        temp_seg_data = temp_seg_data.astype(np.uint32)

        outImg = CXImage()
        strImgPath = r"C:\Users\Admin\Desktop\2019_12_13_normal_200pclassTrainingset_128random_spectrum\P" + str(
            j) + ".tiff"
        outImg.Create(3, xImgIn.m_nLines, xImgIn.m_nSamples, np.float32, strImgPath)
        outImg.WriteImgData(temp_image_data, currentHeight, currentWidth, currentPosX, currentPosY, padding,
                            data_arrange=0)

        outImg_gt = CXImage()
        strImgPath_gt = r"C:\Users\Admin\Desktop\2019_12_13_normal_200pclassTrainingset_128random_spectrumgt\P" + str(
            j) + ".tiff"
        outImg_gt.Create(1, xImgIn.m_nLines, xImgIn.m_nSamples, np.uint32, strImgPath_gt)
        outImg_gt.WriteImgData(temp_seg_data, currentHeight, currentWidth, currentPosX, currentPosY, padding,
                               data_arrange=0)

        # test
        test_label_random_array = generate_array(test_assign, groundTruthData.shape[0], groundTruthData.shape[1])
        test_image_random_array = np.array([test_label_random_array, test_label_random_array])
        temp_test_label_random_array = np.array([test_label_random_array])
        for k in range(1):
            test_image_random_array = np.concatenate((test_image_random_array, temp_test_label_random_array), axis=0)
        test_image_random_array = test_image_random_array.transpose((1, 2, 0))

        # 根据测试集image的mask对imageData进行处理，将忽略的pixel置0
        temp_image_data = selected_inImgData * test_image_random_array
        temp_image_data = temp_image_data.astype(np.float32)
        # 根据测试集label的mask对groundTruthData进行处理，将忽略的pixel置0
        temp_seg_data = groundTruthData * test_label_random_array
        temp_seg_data = temp_seg_data.astype(np.uint32)

        outImg = CXImage()
        strImgPath = r"C:\Users\Admin\Desktop\2019_12_13_normal_200pclassTrainingset_128random_spectrum\P" + str(
            201 + j) + ".tiff"
        outImg.Create(3, xImgIn.m_nLines, xImgIn.m_nSamples, np.float32, strImgPath)
        outImg.WriteImgData(temp_image_data, currentHeight, currentWidth, currentPosX, currentPosY, padding,
                            data_arrange=0)
        del outImg

        outImg_gt = CXImage()
        strImgPath_gt = r"C:\Users\Admin\Desktop\2019_12_13_normal_200pclassTrainingset_128random_spectrumgt\P" + str(
            201 + j) + ".tiff"
        outImg_gt.Create(1, xImgIn.m_nLines, xImgIn.m_nSamples, np.uint32, strImgPath_gt)
        outImg_gt.WriteImgData(temp_seg_data, currentHeight, currentWidth, currentPosX, currentPosY, padding,
                               data_arrange=0)
