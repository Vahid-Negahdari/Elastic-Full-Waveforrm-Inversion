import tensorflow as tf
import numpy as np
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

#########################################################
# Define Hyperparameter
#########################################################

train_epochs = 30
batch_size   = 25
BIGG_BATCH   = 27000
num_batch    = int(BIGG_BATCH/batch_size)
lr           = 0.02
n            = 51
k            = 9
N            = 2*k*102
semi         = int(np.ceil(N/8)*128)

#########################################################
# Import Data
#########################################################
path = Path.cwd() /('Dataset')

if path.is_dir() == False :
   print('Downloading Dataset...')
   url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5ggj5twn75-1.zip'
   http_response = urlopen(url)
   archive = ZipFile(BytesIO(http_response.read()))
   archive.extractall(path=path.parent)
   (path.parent/('LINF_180012400')).rename('Dataset')
   print('Download Completed.')


Density_train = np.load(path / ('Density_Train.npy'), allow_pickle=True)
Density_test  = np.load(path / ('Density_Test.npy'), allow_pickle=True)

Surface_train = np.load(path / ('Surface_Train.npy'), allow_pickle=True)
Surface_test  = np.load(path / ('Surface_Test.npy'), allow_pickle=True)

Surface_train = np.reshape(Surface_train, [BIGG_BATCH, 2*n*k, 1])
Surface_train = np.concatenate((np.real(Surface_train ), np.imag(Surface_train )), axis=1)
Surface_test  = np.reshape(Surface_test, [1000, 2*n*k, 1])
Surface_test  = np.concatenate((np.real(Surface_test ), np.imag(Surface_test )), axis=1)

#########################################################
# Define Some Functions
#########################################################
def conv(input, w, stride):
    y = tf.nn.conv1d(input=input, filters=w, stride=stride,padding='SAME')
    y = tf.nn.leaky_relu(y)
    return y


def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

#########################################################
# Define Weights
#########################################################
def get_tfVariable(shape, name):
    return tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)(shape), name=name, trainable=True, dtype=tf.float32)

weights=[]
weights = weights + [get_tfVariable([3,1,32],   'W0')]
weights = weights + [get_tfVariable([3,32,64],  'W1')]
weights = weights + [get_tfVariable([3,64,128],  'W3')]
weights = weights + [get_tfVariable([semi,n**2],'W5')]
weights = weights + [get_tfVariable([n**2],   'W6')]


########################################################
# Define Model
########################################################
def Model(u):
    C = conv(u, weights[0],2)
    C = conv(C, weights[1],2)
    C = conv(C, weights[2],2)
    C = tf.reshape(C,[C.shape[0],semi])
    C = fullyConnected_layer(C, weights[3], weights[4])
    return C

#########################################################
# Define Loss Function
#########################################################
def loss_function(y_pred, y_true):
     Loss = tf.reduce_mean(tf.square(y_pred- y_true))
     return  Loss

#######################################################
#######################################################
def train_step(u, uu,lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = Model(u)
        current_loss = loss_function(preds, uu)
        grads = tape.gradient(current_loss, weights )
        optimizer.apply_gradients(zip(grads, weights ))
        return current_loss


def Test_Score(epoch):
    preds = Model(Surface_test)
    loss = loss_function(preds, Density_test)
    print("--- On epoch Test {} ---".format(epoch))
    tf.print(" Loss:",loss)
    print("\n")



################################################################
# Training Process
################################################################
for epoch in range(train_epochs):
      avg_Loss = 0
      if np.mod(epoch,2)==0:
         lr=lr/2

      for s in range(num_batch):
          batch_u   = Density_train  [s * batch_size  : (s + 1) * batch_size ]
          batch_uu  = Surface_train[s * batch_size  : (s + 1) * batch_size ]
          Loss      = train_step(batch_uu, batch_u,lr)
          avg_Loss +=  Loss / num_batch
      print("--- On epoch {} ---".format(epoch))
      tf.print(" Loss:", avg_Loss)
      print("\n")
      if (epoch % 3 == 0):
         Test_Score(epoch)


Fake_Dens = np.zeros([28000,n**2]).astype('float32')

Density = np.concatenate((Density_train,Density_test),axis=0)
Surface = np.concatenate((Surface_train,Surface_test),axis=0)
for i in range(28):
    preds = Model(Surface[i*1000: (i+1)*1000]).numpy()
    Fake_Dens[i*1000:(i+1)*1000] = np.reshape(preds, [1000, n ** 2])



d1=np.sqrt(np.sum(np.square(Fake_Dens[0:27000]-Density[0:27000]),axis=1)) / np.sqrt(np.sum(np.square(Density[0:27000]),axis=1))
d2=np.sqrt(np.sum(np.square(Fake_Dens[27000:28000]-Density[27000:28000]),axis=1)) / np.sqrt(np.sum(np.square(Density[27000:28000]),axis=1))
print(np.mean(d1))
print(np.mean(d2))




