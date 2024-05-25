import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path


#########################################################
# Define Hyperparameter
#########################################################

train_epochs   = 23
batch_size     = 25
num_BIGG_BATCH = 27
BIGG_BATCH     = 1000
num_batch      = int(BIGG_BATCH/batch_size)
lr             = 0.01
n              = 51
k              = 9
semi           = int(np.ceil(4*n/16)*256)

path  = Path('/home/cvl/Pycharm/pythonProject/Final/L10K40')
path2 = Path('/home/cvl/Pycharm/pythonProject/Final/L10K40/L2')

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


def get_tfVariable(shape, name):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=14)(shape), name=name, trainable=True, dtype=tf.float32)

########################################################
# Define Model
########################################################

def model( u ):
    C = conv(u, weights[0],2)
    C = conv(C, weights[1],2)
    C = conv(C, weights[2],2)
    C = conv(C, weights[3],2)
    C = tf.reshape(C,[C.shape[0],semi])
    C = fullyConnected_layer(C, weights[4], weights[5])
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
        preds = model(u)
        current_loss = loss_function(preds, uu)
        grads = tape.gradient(current_loss, weights )
        optimizer.apply_gradients(zip(grads, weights ))
        return current_loss ,preds.numpy()


#######################################################
# Training Process
#######################################################

for p in range(4):
    print(p)
    if   p==0:
        name1 = 'Real'     ; name2 = 'H'
    elif  p==1:
        name1 = 'Real'     ; name2 = 'V'
    elif  p==2:
        name1 = 'Complex'  ; name2 = 'H'
    elif  p==3:
        name1 = 'Complex'  ; name2 = 'V'




    weights = []
    weights = weights + [get_tfVariable([3, k, 32], 'W0')]
    weights = weights + [get_tfVariable([3, 32, 64], 'W1')]
    weights = weights + [get_tfVariable([3, 64, 128], 'W3')]
    weights = weights + [get_tfVariable([3, 128, 256], 'W3')]
    weights = weights + [get_tfVariable([semi, k * 1 * n ** 2], 'W5')]
    weights = weights + [get_tfVariable([k * 1 * n ** 2], 'W6')]

    lr = 0.01
    ########################################################
    ################################################################
    for epoch in range(train_epochs):
          avg_Loss = 0
          if np.mod(epoch,2)==0 :
             lr=lr/2
          for j in range(num_BIGG_BATCH):
                if epoch == train_epochs - 1:
                         Fake_sc = np.zeros([BIGG_BATCH, n ** 2, k], dtype=np.float32)
                Surface = np.load(path / ('surface' + str(j) + '.npy'), allow_pickle=True)
                Surface = np.concatenate((np.real(Surface), np.imag(Surface)), axis=1)

                if p==0 or p==2 :
                     Scatt   = np.load(path / ('scatt_'+ name1 + str(j) + '.npy'), allow_pickle=True)[:,0:n**2,:]
                else :
                     Scatt   = np.load(path / ('scatt_'+ name1 + str(j) + '.npy'), allow_pickle=True)[:,n**2:2*(n**2),:]

                Scatt   = np.reshape  (Scatt, [BIGG_BATCH , k*n**2])
                for s in range(num_batch):
                      batch_u    = np.real(Surface[s * batch_size: (s + 1) * batch_size])
                      batch_uu   = Scatt  [s * batch_size  : (s + 1) * batch_size ]

                      Loss ,preds      = train_step(batch_u, batch_uu,lr)
                      avg_Loss +=  Loss / (num_batch*num_BIGG_BATCH)
                      if epoch == train_epochs -1 :
                         Fake_sc[s * batch_size:(s + 1) * batch_size  ] = np.reshape(preds,[batch_size,n**2,k])

                if epoch == train_epochs - 1:
                   np.save(path2 / ('Fake_sc_'+ name1 +'_' +name2 + '_'  + str(j) + '.npy'), Fake_sc)
          print("--- On epoch {} ---".format(epoch))
          tf.print(" Loss:", avg_Loss)
          print("\n")

    Surface = np.load(path / ('surface' + str(27) + '.npy'), allow_pickle=True)
    Surface = np.concatenate((np.real(Surface), np.imag(Surface)), axis=1)
    preds   =  model(Surface)
    preds   = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
    np.save(path2 / ('Fake_sc_' + name1 + '_' + name2 + '_' + str(27) + '.npy'), preds)




del Fake_sc
del Surface
del Scatt

for j in range(num_BIGG_BATCH + 1 ):
    A = np.load(path2 / ('Fake_sc_Real_H' + '_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    B = np.load(path2 / ('Fake_sc_Real_V' + '_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    np.save(path2 / ('Fake_Real' + str(j) + '.npy'),  np.concatenate(  (A, B) , axis=1)     )
    C = np.load(path2 / ('Fake_sc_Complex_H' +'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    D = np.load(path2 / ('Fake_sc_Complex_V' +'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    np.save(path2 / ('Fake_Complex' + str(j) + '.npy'),  np.concatenate(  (C, D) , axis=1)     )