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
    model_.fit(data, labels)
    model_.save(file_name)
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
    model_.add(Dense(2, activation='softmax'))
    model_.compile(optimizer='SGD', loss='categorical_crossentropy')
    return model_


if __name__ == "__main__":
    from cv2 import VideoCapture, imshow, waitKey, resize
    label = (input(
        "Write whether the label to the camera cap is true\n") in ["1", "true", "True", "yes", "Yes", "y", "Y"])
    if label:
        label = [1, 0]
    else:
        label = [0, 1]
    print("Label:", label)

    cap = VideoCapture(0)

    data = []
    labels = []
    for n_pics in range(100):
        frame = cap.read()  # Returns [bool Success, mat img]
        frame = resize(frame[1], (80, 60))

        data.append(frame)
        labels.append(label)

        imshow("Kuva", frame)
        waitKey(1)

    print(n_pics+1, "images taken with all images labelled as", str(label))
    data = array(data)
    labels = array(labels)

    file_name = "saved_model.h5"

    try:
        model = load_model(file_name)
    except OSError:
        print("No saved model found, can not check the accuracy before training.")

    if not load_trainmore_save(file_name, data, labels):
        print("No saved model found, making a new one and training it...")
        model = make_VGGlike_convnet()
        model.fit(data, labels)
        model.save(file_name)
