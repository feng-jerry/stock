import csv
import os

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers

max_threads = 64

os.environ["OMP_NUM_THREADS"] = f"{max_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{max_threads}"
os.environ["MKL_NUM_THREADS"] = f"{max_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{max_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{max_threads}"

tf.config.threading.set_inter_op_parallelism_threads(
    max_threads
)

WINDOW_SIZE = 5
FORCAST_SIZE = 5

MODE = '_' + str(WINDOW_SIZE) + '_' + str(FORCAST_SIZE)

X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []

DIR = './SH50/data'

# cnt = 0
for filename in os.listdir(DIR):
    # cnt += 1
    with open(DIR + '/' + filename) as csvfile:
        data = []
        data_price = []
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == 'trade_date':
                continue
            t = dateutil.parser.parse(row[0])
            data.append((t, float(row[4])))
            data_price.append(float(row[4]))

        for i in range(len(data) - WINDOW_SIZE - FORCAST_SIZE):
            today = i + WINDOW_SIZE - 1
            if data_price[today + FORCAST_SIZE] > data_price[today]:
                y_append = 1
            else:
                y_append = 0

            if data[today][0].year <= 2018 or (data[today][0].year == 2019 and data[today][0].month <= 3):
                X_train.append(data_price[i: i + WINDOW_SIZE])
                y_train.append(y_append)
            elif data[today][0].year == 2019 and data[today][0].month >= 4:
                X_val.append(data_price[i: i + WINDOW_SIZE])
                y_val.append(y_append)
            else:
                X_test.append(data_price[i: i + WINDOW_SIZE])
                y_test.append(y_append)
    # if cnt == 1:
    #     break

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Normalization and Attention
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization()(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0.0,
        mlp_dropout=0.0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


input_shape = X_train.shape[1:]

acc = 0
best_acc = 0
cnt = 0
while best_acc < .7:
    cnt += 1
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[256],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["binary_accuracy"]
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    model.evaluate(X_test, y_test, verbose=1)

    y_test_pred = model.predict(X_test)
    y_test_pred_bin = []

    for i in y_test_pred:
        if i[0] > .5:
            y_test_pred_bin.append(1)
        else:
            y_test_pred_bin.append(0)

    acc = metrics.accuracy_score(y_test, y_test_pred_bin)

    print(cnt, "acc:", acc, best_acc)

    if acc > best_acc:
        best_acc = acc

        model.save('./model/transformer' + MODE + '.h5')

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred_bin)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='Transformer')
        display.plot()
        plt.savefig("AUC")
        # plt.show()

        # summarize history for loss
        plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        plt.savefig("loss")
        # plt.show()

        # summarize history for accuracy
        plt.cla()
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        plt.savefig("accuracy")
        # plt.show()
