import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt
import shutil
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

original_dataset_dir_elephant = '/home/janusz/Downloads/HW5/elephant'
original_dataset_dir_dogs = '/home/janusz/Downloads/HW5/dogs'
original_dataset_dir_cats = '/home/janusz/Downloads/HW5/cats'
if not os.path.exists(original_dataset_dir_elephant):
    print('no elephant data directory')
if not os.path.exists(original_dataset_dir_dogs):
    print('no dog data directory')
if not os.path.exists(original_dataset_dir_cats):
    print('no cat data directory')


    base_dir = '/home/janusz/Downloads/HW5/cats_and_dogs_elephant_small'
if not os.path.exists(base_dir):
    print('creating new directory for small dataset')
    os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    print('creating new train sudirectory for small dataset')
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    print('creating new validation su directory for small dataset')
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    print('creating new test subdirectory for small dataset')
    os.mkdir(test_dir)


train_elephant_dir = os.path.join(train_dir, 'elephant')
if not os.path.exists(train_elephant_dir):
    os.mkdir(train_elephant_dir)

validation_elephant_dir = os.path.join(validation_dir, 'elephant')
if not os.path.exists(validation_elephant_dir):
    os.mkdir(validation_elephant_dir)

test_elephant_dir = os.path.join(test_dir, 'elephant')
if not os.path.exists(test_elephant_dir):
    os.mkdir(test_elephant_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)


test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)




if len(os.listdir(train_elephant_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephant, fname)
        dst = os.path.join(train_elephant_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_elephant_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(100, 200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephant, fname)
        dst = os.path.join(validation_elephant_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_elephant_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(400, 100)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephant, fname)
        dst = os.path.join(test_elephant_dir, fname)
        shutil.copyfile(src, dst)

    if len(os.listdir(train_cats_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_cats_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(100, 200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_cats_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(400, 100)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(train_dogs_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_dogs_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(100, 200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_dogs_dir)) == 0:
    fnames = ['{}.jpg'.format(i) for i in range(400, 100)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)



datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(175, 175),
        batch_size=25,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(175, 175),
        batch_size=25,
        class_mode='binary')

test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(175, 175),
        batch_size=25,
        class_mode='binary')


conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(175, 175, 5))
print(conv_base.summary())



model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))
model.compile(optimizer= tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])



history = model.fit(
      train_generator,
      steps_per_epoch=25,
      epochs=8,
      validation_data=validation_generator,
      validation_steps=10,
      verbose=2)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt
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
plt.clf()



test_loss, test_acc = model.evaluate(test_generator, steps=25)
print('test acc:', test_acc)



def extract_features(directory, sample_count, batch_size = 25):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(175, 175),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        #if i>=38:
        #   print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels



train_features, train_labels = extract_features(train_dir, 1400)
validation_features, validation_labels = extract_features(validation_dir, 200)
test_features, test_labels = extract_features(test_dir, 400)
train_features = np.reshape(train_features, (800, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (400, 4 * 4 * 512))
test_features = np.reshape(test_features, (800, 4 * 4 * 512))



decision_model = models.Sequential()
decision_model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
decision_model.add(layers.Dropout(0.5))
decision_model.add(layers.Dense(1, activation='sigmoid'))

decision_model.compile(optimizer= tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

decision_history = decision_model.fit(train_features, train_labels,
                    epochs=35,
                    batch_size=25,
                    validation_data=(validation_features, validation_labels))



d_acc = decision_history.history['acc']
d_val_acc = decision_history.history['val_acc']
d_loss = decision_history.history['loss']
d_val_loss = decision_history.history['val_loss']

d_epochs = range(len(d_acc))

plt.plot(d_epochs, d_acc, 'bo', label='Training acc')
plt.plot(d_epochs, d_val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(d_epochs, d_loss, 'bo', label='Training loss')
plt.plot(d_epochs, d_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.clf()



d_results = decision_model.evaluate(test_features, test_labels)
print(d_results)



e_to_e_model = models.Sequential()
e_to_e_model.add(conv_base)
e_to_e_model.add(layers.Flatten())
e_to_e_model.add(layers.Dense(256, activation='relu'))
e_to_e_model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = True



set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False



e_to_e_model.compile(loss='binary_crossentropy',
              optimizer= tf.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

e_to_e_history = e_to_e_model.fit(
      train_generator,
      steps_per_epoch=20,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=10,
      verbose=2)



e_acc = e_to_e_history.history['acc']
e_val_acc = e_to_e_history.history['val_acc']
e_loss = e_to_e_history.history['loss']
e_val_loss = e_to_e_history.history['val_loss']

e_epochs = range(len(e_acc))

plt.plot(e_epochs, e_acc, 'bo', label='Training acc')
plt.plot(e_epochs, e_val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(e_epochs, e_loss, 'bo', label='Training loss')
plt.plot(e_epochs, e_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.clf()



e_test_loss, e_test_acc = e_to_e_model.evaluate(test_generator, steps=20)
print('test acc:', e_test_acc)
