import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist,fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

input_layer=layers.Input(shape=(48,48,3))
model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)

last_layer=model_vgg16.output
flatten=layers.Flatten()(last_layer)

# Add dense layer
dense1=layers.Dense(100,activation='relu')(flatten)
dense1=layers.Dense(100,activation='relu')(flatten)
# Add dense layer to the final output layer
output_layer=layers.Dense(10,activation='softmax')(flatten)

# Creating modle with input and output layer
model_m=models.Model(inputs=input_layer,outputs=output_layer)

for layer in model_m.layers[:-1]:
    layer.trainable=False
# Compiling Model
model_m.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
     # Load MNIST dataset
(x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()

# Load and preprocess the MNIST dataset

x_train_m= np.expand_dims(x_train_m, axis=-1)  # Add channel dimension
x_test_m= np.expand_dims(x_test_m, axis=-1)
x_train_m = np.repeat(x_train_m, 3, axis=-1)  # Convert to 3 Channels
x_test_m= np.repeat(x_test_m, 3, axis=-1)
x_train_m = tf.image.resize(x_train_m, (48, 48))  # Resize images to (48, 48)
x_test_m = tf.image.resize(x_test_m, (48, 48))
x_train_m.shape, x_test_m.shape
     
# Normalize pixel values to be between 0 and 1
x_train_m= x_train_m / 255.0
x_test_m = x_test_m / 255.0

# Convert labels to categorical format
y_train_m= to_categorical(y_train_m, 10)
y_test_m = to_categorical(y_test_m, 10)
     
training_m = model_m.fit(x_train_m,y_train_m,epochs=10,batch_size=200,verbose=True,validation_data=(x_test_m,y_test_m))
# Evaluate the model on MNIST testing samples
y_pred_m = model_m.predict(x_test_m)
y_pred_classes_m = np.argmax(y_pred_m, axis=1)
y_true_m = np.argmax(y_test_m, axis=1)

# Compute confusion matrix, precision, recall, and F1 score
confusion_m = confusion_matrix(y_true_m, y_pred_classes_m)
precision_m = precision_score(y_true_m, y_pred_classes_m, average='weighted')
recall_m = recall_score(y_true_m, y_pred_classes_m, average='weighted')
f1_m = f1_score(y_true_m, y_pred_classes_m, average='weighted')

print("Confusion Matrix (MNIST):")
print(confusion_m)
print(f"Precision (MNIST): {precision_m}")
print(f"Recall (MNIST): {recall_m}")
print(f"F1 Score (MNIST): {f1_m}")

model_m.save('numFashion10e.keras')



"""
""
# Creating modle with input and output layer
model_f=models.Model(inputs=input_layer,outputs=output_layer)

for layer in model_f.layers[:-1]:
    layer.trainable=False

# Compiling Model

model_f.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
     

# Load Fashion MNIST dataset
(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()

# Load and preprocess the MNIST dataset

x_train_f= np.expand_dims(x_train_f, axis=-1)  # Add channel dimension
x_test_f= np.expand_dims(x_test_f, axis=-1)
x_train_f = np.repeat(x_train_f, 3, axis=-1)  # Convert to 3 channels image
x_test_f = np.repeat(x_test_f, 3, axis=-1)
x_train_f = tf.image.resize(x_train_f, (48, 48))  # Resize images to (48, 48)
x_test_f = tf.image.resize(x_test_f, (48, 48))
x_train_f.shape, x_test_f.shape
     

# Normalize pixel values to be between 0 and 1
x_train_f= x_train_f / 255.0
x_test_f = x_test_f / 255.0

# Convert labels to categorical format
y_train_f= to_categorical(y_train_f, 10)
y_test_f= to_categorical(y_test_f, 10)
     


training_f = model_f.fit(x_train_f,y_train_f,epochs=2,batch_size=200,verbose=True,validation_data=(x_test_f,y_test_f))
     

# Evaluate the model on Fashion MNIST testing samples
y_pred_f = model_f.predict(x_test_f)
y_pred_classes_f= np.argmax(y_pred_f, axis=1)
y_true_f = np.argmax(y_test_f, axis=1)


# Compute confusion matrix, precision, recall, and F1 score
confusion_f= confusion_matrix(y_true_f, y_pred_classes_f)
precision_f = precision_score(y_true_f y_pred_classes_f, average='weighted')
recall_f = recall_score(y_true_f, y_pred_classes_f, average='weighted')
f1_f = f1_score(y_true_f, y_pred_classes_f, average='weighted')

print("(Fashion MNIST):")
print("Confusion Matrix :")
print(confusion_f)
print(f"Precision : {precision_f}")
print(f"Recall : {recall_f}")
print(f"F1 Score : {f1_f}")
     
"""

     


     
