from warnings import simplefilter

# Ignore FutureWarnings from keras
simplefilter(action='ignore', category=FutureWarning)

from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout

from numpy import array

def load_trainmore_save(file_name, data):
    pass

def make_VGGlike_convnet():
    model_ = Sequential()

    model_.add(Conv2D(16, (3,3), activation='relu', input_shape=(60,80,3,)))
    model_.add(Conv2D(16, (3,3), activation='relu'))
    model_.add(MaxPool2D(pool_size=(2,2)))
    model_.add(Dropout(0.2))

    model_.add(Conv2D(32, (3,3), activation='relu'))
    model_.add(Conv2D(32, (3,3), activation='relu'))
    model_.add(MaxPool2D(pool_size=(2,2)))
    model_.add(Dropout(0.2))

    model_.add(Flatten())
    model_.add(Dense(256, activation='relu'))
    model_.add(Dropout(0.3))
    model_.add(Dense(2, activation='softmax'))
    model_.compile(optimizer='SGD', loss='categorical_crossentropy')
    return model_

if __name__ == "__main__":
    from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY, imshow, waitKey, resize
    label = (input(
        "Write whether the label to the camera cap is true\n") in ["1", "true", "True", "yes", "Yes", "y", "Y"])
    print("Label:", label)

    cap = VideoCapture(0)
    
    data = []
    labels = []
    for n_pics in range(100):
        frame = cap.read()  # Returns [bool Success, mat img]
        frame = resize(frame[1], (80,60))

        data.append(frame)
        if label:
            label_=[1,0]
        else:
            label_=[0,1]
        labels.append(label_)

        imshow("Kuva", frame)
        waitKey(1)
    
    print(n_pics+1, "images taken, training model with all images labelled as", str(label)+"...")

    data = array(data)
    labels = array(labels)
    
    model = make_VGGlike_convnet()
    model.fit(data, labels)
