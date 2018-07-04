# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import numpy as np

count_rand_crop = 30
data_size = 32
label_size = 20
conv_side = 6
scale = 2


def prepare_crop_data(path):
    '''
    выборка из случайных областей изображений
    :param path:
    :return:
    '''
    names = os.listdir(path)
    names = sorted(names)
    nums = len(names)

    data = np.zeros((nums * count_rand_crop, 1, data_size, data_size), dtype=np.float)
    label = np.zeros((nums * count_rand_crop, 1, label_size, label_size), dtype=np.float)

    for i in range(nums):
        name = path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # уменьшение разрешения
        lr_img = cv2.resize(hr_img, (shape[1] / scale, shape[0] / scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # генерация случайних координат для выборки из изображения
        points_x = np.random.randint(0, min(shape[0], shape[1]) - data_size, count_rand_crop)
        points_y = np.random.randint(0, min(shape[0], shape[1]) - data_size, count_rand_crop)

        for j in range(count_rand_crop):
            lr_patch = lr_img[points_x[j]: points_x[j] + data_size, points_y[j]: points_y[j] + data_size]
            hr_patch = hr_img[points_x[j]: points_x[j] + data_size, points_y[j]: points_y[j] + data_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * count_rand_crop + j, 0, :, :] = lr_patch
            label[i * count_rand_crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label


block_step = 16
block_size = 32

def prepare_all_data(path):
    '''
    выборка по всему изображению
    :param path:
    :return:
    '''
    names = os.listdir(path)
    names = sorted(names)
    nums = len(names)

    data = []
    label = []

    for i in range(nums):
        name = path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape

        # уменьшение разрешения
        lr_img = cv2.resize(hr_img, (shape[1] / scale, shape[0] / scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num = (shape[0] - (block_size - block_step) * 2) / block_step
        height_num = (shape[1] - (block_size - block_step) * 2) / block_step
        for k in range(width_num):
            for j in range(height_num):
                x = k * block_step
                y = j * block_step
                hr_patch = hr_img[x: x + block_size, y: y + block_size]
                lr_patch = lr_img[x: x + block_size, y: y + block_size]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = np.zeros((1, data_size, data_size), dtype=float)
                hr = np.zeros((1, label_size, label_size), dtype=float)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                data.append(lr)
                label.append(hr)

    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    return data, label


def write_hdf5(data, labels, output_filename):
    x = data.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


if __name__ == "__main__":
    data_path = "train_HR/"
    test_path = "test_HR/"

    data, label = prepare_all_data(data_path)
    write_hdf5(data, label, "train_HR.h5")
    data, label = prepare_all_data(test_path)
    write_hdf5(data, label, "test_HR.h5")
