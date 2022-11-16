#라이브러리 프레임워크 호출
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#MNIST 데이터셋 호출 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#데이터 정규화 
X_train, X_test = X_train/255.0, X_test/255.0

#이미지 데이터 reshape

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Sequential API 사용해 model 생성
model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu',input_shape=(28,28,1), padding = 'same'),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'),
                             tf.keras.layers.MaxPooling2D((2,2)),

                             tf.keras.layers.Conv2D(128, (3,3), activation='relu',input_shape=(28,28,1), padding = 'same'),
                             tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding = 'valid'),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             
                             #Classifier 출력층
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=512, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=256, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units= 10,activation='softmax')
                             
])

#모델 컴파일 
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#모델 학습 
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=10, batch_size=100)

result = model.evaluate(X_test, y_test)
print("최종 예측 성공률(%): ", result[1]*100)

#모델 저장 
model.save('my_model.h5')

