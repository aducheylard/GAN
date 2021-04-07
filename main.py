import tensorflow as tf
from tensorflow.python.client import device_lib 
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import tensorflow_docs.vis.embed as embed

#Modelo del generador
def make_generator_model():
  model = tf.keras.Sequential()
  #agremos la capa incial sin bias, como input recive un ndarray de 100 (del ruido generado) y la primera hidden layer para que entregue un resultado con 7*7*256
  model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization()) # Normaliza sus inputs manteniendo el promedio del output y desviacion estandar a 0 y 1 respectivamente.
  model.add(layers.LeakyReLU()) #Funcion de activacion, se usa la LeakyReLU ya que modifica la funcion original para que enves de retornar 0 cuando su input es negativo, retorna 0.01, de esta forma, la neurona no queda completamente desactivada.

  model.add(layers.Reshape((7, 7, 256))) #Se transforma el output de la capa anterior a las dimension de 7,7,256
  assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size....esto asegura que las dimension son de 7,7,256. Sino, erroja una excepcion

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) # Se agrega la capa DesConvolucional o Convolucional Transpuesta, que consiste en ser una Convolucional pero en direccion opuesta. Tamano del kernel de 5x5, con un padding de 'same' para que el input tenga el mismo padding que el output en todas sus direcciones
  assert model.output_shape == (None, 7, 7, 128)  # Note: None is the batch size....esto asegura que las dimension son de 7,7,128. Sino, erroja una excepcion
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64) # Note: None is the batch size....esto asegura que las dimension son de 14,14,64. Sino, erroja una excepcion
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) #output layer del modelo, usando la funcion de activacion de 'tanh', eso retorna un valor entre -1 y 1
  assert model.output_shape == (None, 28, 28, 1)# Note: None is the batch size....esto asegura que las dimension son de 28,28,1. Sino, erroja una excepcion

  return model

def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

#La funcion de perdida del discriminador identifica que tan bien es capaz de distinguir entre imagenes reales o falsas.
#compara las predicciones del discriminador de las imagenes reales con un arreglo de 1's, y las falsas con 0's
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

#La funcion de perdida del generador quantifica que tan bien pudo enganar al discriminador
#cuando el generador funciona bien, el discrtiminador va a clasificar las imagenes falsas (o creadas) como reales.
#se compara las deciciones del discriminador sobre las imagenes creadas contra un arreglo de 1's
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

#El entranamiento funciona cuando recive la seed aleatoria como input
#Luego, se crea una imagen a partir de la seed.
#Ahora el discriminador debe clasificar las imagenes relaes del dataset vs las generadas
#Despues se calcula la funcion de perdida de cada modelo, y se usan las gradientes para actualizar los modelos.
#'tf.function' se usa para dejar la funcion como un grafo y pueda ser llamado. Es una forma de optimizar la funcion segun Tensorflow, pero que no deberia usarse en produccion ya que reduce la facilidad de debug.
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True) #genera la imagen falsa

    real_output = discriminator(images, training=True) #retorna la clasificacion de la imagen real
    fake_output = discriminator(generated_images, training=True)#retorna la clasificacion de la imagen falsa

    gen_loss = generator_loss(fake_output) #calcula la funcion de perdida del generador
    disc_loss = discriminator_loss(real_output, fake_output)#calcula la funcion de perdida del discriminador

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) #calcula la gradiente del generador
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables) #calcula la gradiente del discriminador

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) #optimiza el generador dada la perdida de gradiente anterior
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) #optimiza el discriminador dada la perdida de gradiente anterior

def train(dataset, epochs): #funcion de entrenamiento
  for epoch in range(epochs): #por cada epoca
    start = time.time() #se calcula el tiempo transcurrido

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,epoch + 1,seed) #se crea una imagen para el gif

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,epochs,seed)

#Generate and save images
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm). De esta forma, las 'muestras' o 'batches' de los datos quedan normalizados
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



#Check de versoin de tf y GPU's
#print(device_lib.list_local_devices())
#print(tf.__version__)


#Cargamos el dataset de MNIST para entrenar al generador y discriminador.
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

#Redimensionamos la imagen a 28x28px de 1 canal...Blanco y negro o escala de grises
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#Normalizamos las imagenes a -1,1 para que coincidan con el rango de la funcion de activacion de tanh
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256 #tamano de las muestras...128, 256, etc

# Se separa la data segun lo definido anteriormente, y se revuelven los datos
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#creamos el modelo del generador sin ningun entrenamiento para generar las imagenes
generator = make_generator_model()

#generamos el ruido o data aleatoria
noise = tf.random.normal([1, 100])
#Le decimos al modelo generador que use el ruido creado anteriormente como input
generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')

#se usa el discriminador sin entrenar para clasificar las imagenes generadas, diciendo si son reales o falsas
#El modelo se entrenara considerando valores positivos para las imagenes reales y negativos para las falsas.
discriminator = make_discriminator_model()
decision = discriminator(generated_image)

print(decision) #ejemplo.........tf.Tensor([[-0.00182554]], shape=(1, 1), dtype=float32).........como es negativo detecta que la imagen es falsa.

#Define the loss and optimizers
#Define loss functions and optimizers for both models.
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Los optimizadores para cada modelo son independientes...porque son 2 modelos distintos.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#Se guardan checkpoints del modelo por si se interrumpe el proceso
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)

#Parametros para definir el entrenamiento
EPOCHS = 50
noise_dim = 100 #dimension de la data de ruido
num_examples_to_generate = 16 #generar 16 ejemplos

#Reutilizamos la seed para generar la data de ruido, asi es mas facil ver el gif del resultado final
seed = tf.random.normal([num_examples_to_generate, noise_dim])

#Entrenamos ambos modelos
train(train_dataset, EPOCHS)
#restauramos el ultimo checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#display_image(EPOCHS)

#Aqui se crea el gif final de las imagenes guardadas del entrenamiento
#Use imageio to create an animated gif using the images saved during training.
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

embed.embed_file(anim_file)