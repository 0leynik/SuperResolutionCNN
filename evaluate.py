# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D, Input
from keras.optimizers import Adam
import numpy as np
import os
import cv2


def predict_model():

    inputs = Input(shape=(None, None, 1))

    conv = Conv2D(128, (9, 9), padding='valid',activation='relu', input_shape=(None, None, 1))(inputs)
    conv = Conv2D(64, (3, 3), padding='same', activation='relu')(conv)
    outputs = Conv2D(1, (5, 5), padding='valid', activation='linear')(conv)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error')

    return model


def predict_2x(img_path):
    '''
    Используется для оценки качества изображений,
    полученных при помощи бикубической интерполяции
    и при использовании сверточной нейронной сети.

    Исходное изображение сжимается в 2 раза по ширине и высоте,
    потом восстанавливается до исходного размера при помощи кубической интерполяции
    и подается на вход нейросети.

    Далее оценивается качество между полученными ихображениями и оригиналом.
    В качестве метрики сравнения используется PSNR.
    https://ru.wikipedia.org/wiki/Пиковое_отношение_сигнала_к_шуму
    :param img_path:
    :return:
    '''
    if not os.path.isfile(img_path):
        print('Укажите путь до изображения! Ошибка: ' + img_path)
        return

    srcnn_model = predict_model()
    srcnn_model.load_weights(saved_weights_path)

    img_dir = os.path.dirname(img_path)

    img_name = os.path.splitext(os.path.basename(img_path))
    name = img_name[0]
    ext = img_name[1]

    downresized_img_path = os.path.join(img_dir, name + '_downresized' + ext)
    bicubic_img_path = os.path.join(img_dir, name + '_bicubic' + ext)
    srcnn_img_path = os.path.join(img_dir, name + '_srcnn' + ext)


    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    shape = img.shape

    croped_img = cv2.resize(img, (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    cv2.imwrite(downresized_img_path, croped_img)


    YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_img = cv2.resize(YCrCb_img[:, :, 0], (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    YCrCb_img[:, :, 0] = Y_img
    BGR_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(bicubic_img_path, BGR_img)


    Y_img = YCrCb_img[:, :, 0]
    Y_img = cv2.copyMakeBorder(Y_img, 6, 6, 6, 6, cv2.BORDER_REFLECT_101)

    Y_in = np.zeros((1, Y_img.shape[0], Y_img.shape[1], 1), dtype=float)
    Y_in[0, :, :, 0] = Y_img.astype(float) / 255.

    Y_pred = srcnn_model.predict(Y_in, batch_size=1) * 255.

    Y_pred = np.clip(Y_pred, 0, 255)
    Y_pred = Y_pred.astype(np.uint8)
    YCrCb_img[:, :, 0] = Y_pred[0, :, :, 0]
    BGR_pred_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(srcnn_img_path, BGR_pred_img)

    # psnr calculation:
    img_1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2YCrCb)[..., 0]

    img_2 = cv2.imread(bicubic_img_path, cv2.IMREAD_COLOR)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2YCrCb)[..., 0]

    img_3 = cv2.imread(srcnn_img_path, cv2.IMREAD_COLOR)
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2YCrCb)[..., 0]

    print('Metrics \"' + img_path + '\"')
    print('Bicubic: ' + str(cv2.PSNR(img_1, img_2)))
    print('SRCNN: ' + str(cv2.PSNR(img_1, img_3)))
    print('\n')


def increase_2x(img_path):
    '''
    Увеличение исходного изображения в 2 раза по ширине и высоте.
    :param img_path:
    :return:
    '''
    if not os.path.isfile(img_path):
        print('Укажите путь до изображения! Ошибка: ' + img_path)

    srcnn_model = predict_model()
    srcnn_model.load_weights(saved_weights_path)

    img_dir = os.path.dirname(img_path)

    img_name = os.path.splitext(os.path.basename(img_path))
    name = img_name[0]
    ext = img_name[1]

    bicubic_img_path = os.path.join(img_dir, name + '_bicubic' + ext)
    srcnn_img_path = os.path.join(img_dir, name + '_srcnn' + ext)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    shape = img.shape

    YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCrCb_img = cv2.resize(YCrCb_img, (shape[1] * 2, shape[0] * 2), cv2.INTER_CUBIC)
    BGR_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(bicubic_img_path, BGR_img)

    Y_img = YCrCb_img[:, :, 0]
    Y_img = cv2.copyMakeBorder(Y_img, 6, 6, 6, 6, cv2.BORDER_REFLECT_101)

    Y_in = np.zeros((1, Y_img.shape[0], Y_img.shape[1], 1), dtype=float)
    Y_in[0, :, :, 0] = Y_img.astype(float) / 255.

    Y_pred = srcnn_model.predict(Y_in, batch_size=1) * 255.

    Y_pred = np.clip(Y_pred, 0, 255)
    Y_pred = Y_pred.astype(np.uint8)
    YCrCb_img[:, :, 0] = Y_pred[0, :, :, 0]
    BGR_pred_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(srcnn_img_path, BGR_pred_img)


# saved_weights_path = 'weights_loss_HR/iter_1000.h5'
saved_weights_path = 'weights_loss_HR/iter_81000.h5'

if __name__ == "__main__":
    # increase_2x('examples/1.jpg')
    # increase_2x('examples/1.jpg')

    predict_2x('examples/1.jpg')
    predict_2x('examples/3.png')
    predict_2x('examples/butterfly.png')
    predict_2x('examples/0140.png')
    predict_2x('examples/0826.png')
