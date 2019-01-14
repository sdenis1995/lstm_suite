import numpy as np
from keras.models import save_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import CuDNNLSTM
from keras.optimizers import Adam
from keras import metrics
from keras import utils
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import os

class baseModel:
    def __init__(self, series_length, feature_count):
        self.series_length = series_length
        self.feature_count = feature_count
    # transforms time series of different length into time series of series_length
    def transform(self, time_series, dtype=None):
        result = []
        for series in time_series:
            result.extend([series[i-self.series_length+1:i+1] for i in range(self.series_length-1, len(series))])
        return np.array(result, dtype=np.float if dtype is None else dtype)

    def transform_Y(self, time_series, dtype=None):
        result = []
        for series in time_series:
            result.extend(series[self.series_length-1:])
        return np.array(result, dtype=np.int if dtype is None else dtype)

    def base_lstm_model(self):
        # model description, change the architecture the way you want (LSTM and CuDNNLSTM of Keras are interchangeable)
        model = Sequential()
        model.add(CuDNNLSTM(50, return_sequences=True, input_shape=(self.series_length, self.feature_count)))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(25, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(10))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01),
                           metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])
        return model

class LSTM(baseModel):
    # series_length - the amount of points prior to the point of classification
    # char_count - the amount of features, describing each point
    def __init__(self, series_length, feature_count, LSTM_model=None):
        super(LSTM, self).__init__(series_length, feature_count)
        if LSTM_model is None:
            self.model = self.base_lstm_model()
        else:
            self.model = LSTM_model

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    # X - list of time series of different length (in our case - representing different applications)
    def transform_predict(self, X, dtype=None):
        X_transform = self.transform(X, dtype=dtype)
        return self.predict(X_transform)

    def fit(self, X, Y, test_size=None, epochs=1, batch_size=512, verbose=0):

        # we have non-balanced classes, thus we compute weights for the classes
        counts = [0, 0, 0]
        weights = [0, 0, 0]
        for i in range(3):
            counts[i] = len([x for x in Y if x == i])

        for i in range(3):
            weights[i] = np.max(counts) / 3 / counts[i]

        # transform the label vector, so instead of a number each sample is represented by
        # a vector, where the corresponding to a label place 1 is put (0 elsewhere)
        Y = utils.to_categorical(Y)

        if test_size is not None and test_size > 0 and test_size < 1:
            # the division into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
            history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                                batch_size=batch_size, verbose=verbose, shuffle=True, class_weight=weights)
        else:
            history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                shuffle=True, class_weight=weights)

        return history

    # X, Y - list of time series of different length (in our case, they are different labelled applications data)
    def transform_fit(self, X, Y, test_size=None, epochs=1, batch_size=512, verbose=0):
        X_transformed = self.transform(X)
        Y_transformed = self.transform_Y(Y, dtype=np.int)

        indices = [i for i in range(len(Y_transformed)) if Y_transformed[i] >= 0]
        X_transformed = X_transformed[indices]
        Y_transformed = Y_transformed[indices]

        return self.fit(X_transformed, Y_transformed, test_size=test_size, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save_model(self, folder):
        import pickle
        # saving LSTM model
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_model(self.model, "{}/LSTM.h5".format(folder))
        #saving state of the class
        with open('./{}/properties.pickle'.format(folder), 'wb') as handle:
            pickle.dump([self.series_length, self.feature_count], handle)

    @staticmethod
    def load_model(folder):
        LSTM_model = load_model("{}/LSTM.h5".format(folder))
        import pickle
        with open('{}/properties.pickle'.format(folder), 'rb') as handle:
            static_data = pickle.load(handle)
        return LSTM(static_data[0], static_data[1], LSTM_model=LSTM_model)


class StackedLSTM(baseModel):
    def __init__(self, series_length, feature_count, LSTM1=None, LSTM2=None, GDB=None):
        super(StackedLSTM, self).__init__(series_length, feature_count)
        if LSTM1 is None:
            self.LSTM1 = self.base_lstm_model()
        else:
            self.LSTM1 = LSTM1

        if LSTM2 is None:
            self.LSTM2 = self.base_lstm_model()
        else:
            self.LSTM2 = LSTM2

        if GDB is None:
            self.GDB = GradientBoostingClassifier(n_estimators=256)
        else:
            self.GDB = GDB

    def get_LSTM_predictions(self, X):
        pred1 = self.LSTM1.predict(X)
        pred2 = self.LSTM2.predict(X)
        pred = np.concatenate((pred1, pred2), axis=1)
        return pred

    def predict(self, X):
        pred = self.get_LSTM_predictions(X)
        return self.GDB.predict(pred)

    def split_for_fit(self, X, Y):
        # splitting data for training,
        # returns array of 3 lists (contains data and labels), each corresponding for specific classifier (2 LSTM and GDB)
        # change the implementation to fit your problem solution
        result = [[[], []], [[], []], [[], []]]
        for cl in range(3):
            #LSTM1
            result[0][0].extend([X[i] for i in range(len(X)) if Y[i] == cl][::5])
            result[0][0].extend([X[i] for i in range(len(X)) if Y[i] == cl][2::5])
            result[0][1].extend([Y[i] for i in range(len(Y)) if Y[i] == cl][::5])
            result[0][1].extend([Y[i] for i in range(len(Y)) if Y[i] == cl][2::5])
            #LSTM2
            result[1][0].extend([X[i] for i in range(len(X)) if Y[i] == cl][1::5])
            result[1][0].extend([X[i] for i in range(len(X)) if Y[i] == cl][3::5])
            result[1][1].extend([Y[i] for i in range(len(Y)) if Y[i] == cl][1::5])
            result[1][1].extend([Y[i] for i in range(len(Y)) if Y[i] == cl][3::5])
            #GDB1
            result[2][0].extend([X[i] for i in range(len(X)) if Y[i] == cl][4::5])
            result[2][1].extend([Y[i] for i in range(len(Y)) if Y[i] == cl][4::5])
        for i in range(3):
            result[i][0] = np.array(result[i][0])
            result[i][1] = np.array(result[i][1], dtype=np.int)
        return result

    # X - list of time series of different length (in our case - representing different applications)
    def transform_predict(self, X, dtype=None):
        X_transform = self.transform(X, dtype=dtype)
        return self.predict(X_transform)

    def fit(self, X, Y, epochs=1, batch_size=512, test_size=0.05, verbose=0):
        training_data = self.split_for_fit(X, Y)

        # we have non-balanced classes, thus we compute weights for the classes
        counts = [0, 0, 0]
        weights = [0, 0, 0]
        for i in range(3):
            counts[i] = len([x for x in Y if x == i])

        for i in range(3):
            weights[i] = np.max(counts) / 3 / counts[i]

        ### training first LSTM model ###
        # transform the label vector, so instead of a number each sample is represented by
        # a vector, where the corresponding to a label place 1 is put (0 elsewhere)
        print(training_data[0][0].shape)
        print(training_data[0][1].shape)
        print(test_size)
        training_data[0][1] = utils.to_categorical(training_data[0][1])
        if test_size is not None and test_size > 0 and test_size < 1:
            # the division into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(training_data[0][0], training_data[0][1],
                                                                test_size=test_size)
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            self.LSTM1.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, verbose=verbose, shuffle=True, class_weight=weights)
        else:
            self.LSTM1.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True, class_weight=weights)

        ### training second LSTM model ###
        # transform the label vector, so instead of a number each sample is represented by
        # a vector, where the corresponding to a label place 1 is put (0 elsewhere)
        training_data[1][1] = utils.to_categorical(training_data[1][1])
        if test_size is not None and test_size > 0 and test_size < 1:
            # the division into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(training_data[1][0], training_data[1][1],
                                                                test_size=test_size)
            self.LSTM2.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                                batch_size=batch_size, verbose=verbose, shuffle=True, class_weight=weights)
        else:
            self.LSTM2.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                shuffle=True, class_weight=weights)

        ### training GDB ###
        pred = self.get_LSTM_predictions(training_data[2][0])
        self.GDB.fit(pred, training_data[2][1])

    def transform_fit(self,  X, Y, epochs=1, batch_size=512, test_size=0.05, verbose=0):
        X_transformed = self.transform(X)
        Y_transformed = self.transform_Y(Y, dtype=np.int)

        indices = [i for i in range(len(Y_transformed)) if Y_transformed[i] >= 0]
        X_transformed = X_transformed[indices]
        Y_transformed = Y_transformed[indices]

        return self.fit(X_transformed, Y_transformed, test_size=test_size, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_model(self.LSTM1, "{}/LSTM1.h5".format(folder))
        save_model(self.LSTM2, "{}/LSTM2.h5".format(folder))
        import pickle
        with open('{}/stacked.pickle'.format(folder), 'wb') as handle:
            pickle.dump([self.GDB, self.series_length, self.feature_count], handle)

    @staticmethod
    def load_model(folder):
        LSTM1 = load_model("{}/LSTM1.h5".format(folder))
        LSTM2 = load_model("{}/LSTM2.h5".format(folder))
        import pickle
        with open('{}/stacked.pickle'.format(folder), 'rb') as handle:
            static_data = pickle.load(handle)
        return StackedLSTM(static_data[1], static_data[2], LSTM1=LSTM1, LSTM2=LSTM2, GDB=static_data[0])
