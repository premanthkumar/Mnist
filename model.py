import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

#loading dataset
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

#scaling data
x_train = x_train/255
x_test = x_test/255

#model creating
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ["accuracy"]
)

#Training
model.fit(x_train,y_train,epochs=10)

#predictions
y_predictions = model.predict(x_test)
y_outputs = np.math.argmax(y_predictions)
print(y_outputs[0])


#Visiualizing first data on test data
plt.matshow(x_test[0])
plt.show()