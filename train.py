# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import prepare_data as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

def model():
    inputs = Input(shape=(32, 32, 1))

    conv = Conv2D(128, (9, 9), padding='valid', activation='relu', input_shape=(None, None, 1))(inputs)
    conv = Conv2D(64, (3, 3), padding='same', activation='relu')(conv)
    outputs = Conv2D(1, (5, 5), padding='valid', activation='linear')(conv)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error')

    return model


loss_save_dir = 'weights_loss_HR'
if not os.path.isdir(loss_save_dir):
    os.makedirs(loss_save_dir)


def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data, label = pd.read_training_data("train_HR.h5")
    val_data, val_label = pd.read_training_data("test_HR.h5")
    print('data.shape = ' +str(data.shape))

    train_on_batch = True
    if train_on_batch:
        f = open(os.path.join(loss_save_dir, 'metrics.csv'), 'w')

        train_ids = np.arange(0, len(data))

        iter_num = 0
        epoches = 10
        batch_size = 128
        save_model_step = 1000

        for e in range(1, epoches+1):
            print('epoch ' + str(e))
            np.random.shuffle(train_ids)
            train_batchs_ids = np.array_split(train_ids, int((len(data)/batch_size) + 1))

            for tr_batch_ids in train_batchs_ids:
                train_batch_loss = srcnn_model.train_on_batch(data[tr_batch_ids], label[tr_batch_ids])

                val_batch_ids = np.random.choice(len(val_data), batch_size)
                val_batch_loss = srcnn_model.test_on_batch(val_data[val_batch_ids], val_label[val_batch_ids])

                iter_num += 1

                print(str(datetime.now())+' iter ' + str(iter_num) + ', loss: ' + str(train_batch_loss) + ', val_loss: ' + str(val_batch_loss))

                metrics = str(iter_num) + ',' + str(train_batch_loss) + ',' + str(val_batch_loss)
                f.write(metrics)
                f.write('\n')
                f.flush()

                if (iter_num % save_model_step) == 0:
                    srcnn_model.save(os.path.join(loss_save_dir, 'iter_' + str(iter_num) + '.h5'))

        f.close()
    else:
        filepath_val_loss = os.path.join(loss_save_dir, 'val_loss_e_{epoch:02d}_loss_{val_loss:.8f}.h5')
        checkpoint_val_loss = ModelCheckpoint(filepath_val_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label), callbacks=[checkpoint_val_loss], shuffle=True, nb_epoch=200, verbose=1)
        # srcnn_model.load_weights("m_model_adam.h5")

        print(history.history.keys())

        f = open(os.path.join(loss_save_dir, 'metrics.csv'), 'w')
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        for i in range(len(loss)):
            f.write(str(i+1)+','+str(loss[i])+','+str(val_loss[i]))
            f.write('\n')
            f.flush()
        f.close()


def plot_graph():

    metrics = np.loadtxt(os.path.join(loss_save_dir, 'metrics.csv'), delimiter=',')

    mpl.rcParams['figure.figsize'] = [8.4, 4.8]
    mpl.rcParams['figure.dpi'] = 500
    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['axes.linewidth'] = 0.3

    # plot graph loss
    plt.figure('Graph loss')
    plt.title('Loss')
    plt.plot(metrics[:, 1])
    plt.plot(metrics[:, 2])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.grid(True, linestyle='--')
    # plt.yticks(np.linspace(0., 0.2, 11))
    # plt.ylim(0., 0.2)
    # plt.xticks(np.linspace(0, 5000, 11))
    # plt.xlim(0, 5000)
    plt.savefig(os.path.join(loss_save_dir, 'graph_history_loss.png'))

if __name__ == "__main__":
    # train()
    plot_graph()
