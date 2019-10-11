from warnings import simplefilter

# Ignore FutureWarnings from keras
simplefilter(action='ignore', category=FutureWarning)

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout

from numpy import array


def load_trainmore_save(file_name, data, labels):
    try:
        model_ = load_model(file_name)
    except OSError:
        return False
    print("File loaded, training...")
    model_.fit(data, labels, epochs=2)
    model_.save(file_name)
    print("File saved as", file_name)
    return True


def make_VGGlike_convnet():
    model_ = Sequential()

    model_.add(Conv2D(16, (3, 3), activation='relu', input_shape=(60, 80, 3,)))
    model_.add(Conv2D(16, (3, 3), activation='relu'))
    model_.add(MaxPool2D(pool_size=(2, 2)))
    model_.add(Dropout(0.2))

    model_.add(Conv2D(32, (3, 3), activation='relu'))
    model_.add(Conv2D(32, (3, 3), activation='relu'))
    model_.add(MaxPool2D(pool_size=(2, 2)))
    model_.add(Dropout(0.2))

    model_.add(Flatten())
    model_.add(Dense(256, activation='relu'))
    model_.add(Dropout(0.3))
    model_.add(Dense(1, activation='sigmoid'))
    model_.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model_


if __name__ == "__main__":
    from cv2 import VideoCapture, imshow, waitKey, resize
    cap = VideoCapture(0)

    data = []
    labels = []

    for i in range(2):
        input("Take for label " + str(i) + "\nPress ENTER to take")
        for n_pics in range(100):
            frame = cap.read()  # Returns [bool Success, mat img]
            frame = resize(frame[1], (80, 60))

            data.append(frame)
            labels.append(i)

            imshow("Kuva", frame)
            waitKey(1)

    cap.release()

    data = array(data)
    labels = array(labels)

    file_name = "saved_model.h5"

    try:
        model = load_model(file_name)
        print("Saved model found, evaluating current accuracy...")
        preds = model.predict(data)
        accuracy_0 = 0
        avg_0 = 0
        accuracy_1 = 0
        avg_1 = 0
        for pred_i in range(len(preds)):
            if pred_i >= n_pics:
                if preds[pred_i][0] >= 0.5:
                    accuracy_1 += 1
                    avg_1 += preds[pred_i][0]
            else:
                if preds[pred_i][0] < 0.5:
                    accuracy_0 += 1
                    avg_0 += preds[pred_i][0]
        print("Percentage correct:\nLabel 0:", accuracy_0, "\nLabel 1:", accuracy_1, "\nAverages:\nLabel 0:", avg_0/100, "\nLabel 1:", avg_1/100)
    except OSError:
        print("No saved model found, can not check the accuracy before training.")

    if not load_trainmore_save(file_name, data, labels):
        print("No saved model found, making a new one and training it...")
        model = make_VGGlike_convnet()
        model.fit(data, labels, epochs=2)
        model.save(file_name)
