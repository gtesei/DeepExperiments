from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

def extract_features(directory, sample_count, conv_base, x=4, y=4, z=512, sx=150, sy=150):
    features = np.zeros(shape=(sample_count, x, y, z))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,target_size=(sx, sy),batch_size=batch_size,class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

def get_model_no_aug():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',
                  metrics=['acc'])
    return model

def get_model_aug(conv_base):
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    print('This is the number of trainable weights '
          'before freezing the conv base:', len(model.trainable_weights))
    conv_base.trainable = False ### <<<<<<<<<<<<<
    print('This is the number of trainable weights '
          'after freezing the conv base:', len(model.trainable_weights))
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])
    return model

def data_generator_aug(train_dir,validation_dir):
    train_datagen = ImageDataGenerator(
              rescale=1./255,
              rotation_range=40,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_dir,
                # All images will be resized to 150x150
                target_size=(150, 150),
                batch_size=20,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=20,
                class_mode='binary')
    return train_generator , validation_generator

def do_plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


######## MAIN
DO_PLOT = False
base_dir = 'cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

# Here's the list of image classification models (all pre-trained on the ImageNet dataset) that are available as part of keras.applications:
# Xception
# InceptionV3
# ResNet50
# VGG16
# VGG19
# MobileNet
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
print(conv_base)

### FEATURE EXTRACTION WITHOUT DATA AUGMENTATION
print(">>> FEATURE EXTRACTION WITHOUT DATA AUGMENTATION ")
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_features, train_labels = extract_features(train_dir, 2000,conv_base)
validation_features, validation_labels = extract_features(validation_dir, 1000,conv_base)
test_features, test_labels = extract_features(test_dir, 1000,conv_base)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = get_model_no_aug()

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

if DO_PLOT:
    do_plot(history)

### FEATURE EXTRACTION WITH DATA AUGMENTATION
print(">>> FEATURE EXTRACTION WITH DATA AUGMENTATION")
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

model = get_model_aug(conv_base)

train_generator , validation_generator = data_generator_aug(train_dir,validation_dir)

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)

model.save('cats_and_dogs_small_3.h5')

if DO_PLOT:
    do_plot(history)

### FINE-TUNING
print(">>> FINE-TUNING")
print("> BEFORE")
print(conv_base.summary())
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

print("> AFTER")
print(conv_base.summary())

history = model.fit_generator(
              train_generator,
              steps_per_epoch=100,
              epochs=100,
              validation_data=validation_generator,
              validation_steps=50)

model.save('cats_and_dogs_small_4.h5')
if DO_PLOT:
    do_plot(history)






