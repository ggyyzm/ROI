import numpy as np

# 根据一维index输出shape为(len(index),2)的二维矩阵，表示点的横纵坐标
def indexToAssignment(index_, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col
        assign_1 = value % Col
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


# 根据groundTruth和比例，对每个类别分别按比例划分训练集和测试集(stochastic stratified sample)
def sampling(proptionVal, groundTruth):              # divide dataset into train and test datasets
    '''
        proptionVal: 比例
        groundTruth: groundtruth
    '''
    # labels_loc = {}
    train = {}
    test = {}
    m = int(groundTruth.max())
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        # labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))    # 得到此类别测试集的个数
        train[i] = indices[:-nb_val]    # 倒数第nb_val个之前的样本用作训练
        test[i] = indices[-nb_val:]     # 倒数nb_val个样本用作测试
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        # whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


# 根据groundTruth将原数据集按每类别固定num个数量进行划分
def sampling_by_category(num, groundTruth):
    '''
        num：训练集中每个类别的个数
        groundTruth: groundtruth
    '''
    train = {}
    test = {}
    m = int(groundTruth.max())
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        # labels_loc[i] = indices
        train[i] = indices[:num]  # 第num个之前的所有样本用作训练
        test[i] = indices[num:]  # 第num个样本之后的所有样本用作测试
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        # whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


# 生成mask
def generate_array(assign, height, width):
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(assign)):
        result[assign[i][0], assign[i][1]] = 1
    return result


# 根据groundTruth将assign(e.g.训练集)按类别划分
def divide_by_category(assign, groundTruth, num_classes):
    result = {}
    num = np.zeros(num_classes+1, dtype=np.uint32)
    for j in range(1, num_classes+1):
        result[j] = {}

    for i in range(len(assign)):
        # if groundTruth[assign[i][0]][assign[i][1]] != 0:
        result[groundTruth[assign[i][0]][assign[i][1]]][num[groundTruth[assign[i][0]][assign[i][1]]]] = assign[i]
        num[groundTruth[assign[i][0]][assign[i][1]]] += 1
    return result


# 对image进行z-score标准化
def z_score_normalization(image, dtype, data_arrange):
    '''
        data_arrange == 0 <==> (height, width, band)
                     == 1 <==> (band, height, width)(gdal)
    '''
    image = image.astype(np.float32)
    if data_arrange == 0:
        for m in range(image.shape[2]):
            img_mean = np.mean(image[:, :, m])
            img_std = np.std(image[:, :, m])
            image[:, :, m] = (image[:, :, m] - img_mean) / img_std
    elif data_arrange == 1:
        for m in range(image.shape[0]):
            img_mean = np.mean(image[m, :, :])
            img_std = np.std(image[m, :, :])
            image[m, :, :] = (image[m, :, :] - img_mean) / img_std
    else:
        raise AttributeError("data_arrange error!")

    image = image.astype(dtype)
    return image
