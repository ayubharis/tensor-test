import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
tf.enable_eager_execution()
TYPE_NAMES = np.array(['Bug', 'Dark', 'Dragon', 'Electric', 'Fighting', 'Fire', 'Flying', 'Ghost',
  'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'])
TN = {k:v for (k,v) in izip(['Bug', 'Dark', 'Dragon', 'Electric', 'Fighting', 'Fire', 'Flying', 'Ghost',
  'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'], range(17))}
f = open("types.csv", "r")
mapperx, mappery, pkmn = [], [], []
for row in f:
  xs = row.split(',')
  pkmn.append(xs[0])
  mapperx.append(np.array(list(Image.open('pics/'+xs[0]+'.png').convert("RGB").getdata()))/255.0)
  mappery.append(TN[xs[1]])
c = list(zip(mapperx, mappery, pkmn))
random.shuffle(c)
mapperx, mappery, pkmn = zip(*c)
mapperx, mappery = np.array(mapperx), np.array(mappery)
testDatax, trainDatax = mapperx[:49], mapperx[49:]
testDatay, trainDatay = mappery[:49], mappery[49:]
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape = (9216, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(17)
])
predictions = model(trainDatax[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(trainDatax, trainDatay, epochs=15)
model.evaluate(testDatax, testDatay, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
pm = probability_model(testDatax[:10])
for i in range(10):
  """img = mpimg.imread('pics/'+pkmn[i]+'.png')
  plt.title(TYPE_NAMES[np.argmax(pm[i])])
  plt.imshow(img)
  plt.show()"""
  print(pkmn[i], TYPE_NAMES[np.argmax(pm[i])])