from tensorflow import keras
from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Dense, Flatten, Dropout, InputLayer, Input



def get_cnn_model():
    i_layer = Input(shape=(28, 28, 1))
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(i_layer)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(units=264, activation='relu')(x)
    x = Dropout(0.2)(x)

    o_layer = Dense(units=10, activation='softmax')(x)

    model = Model(i_layer, o_layer)
    return model