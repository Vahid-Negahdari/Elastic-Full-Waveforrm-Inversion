import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from pathlib import Path


#########################################################
# Define Hyperparameter
#########################################################
train_epochs    = 7
train_epochs1   = 12
train_epochs2   = 4
batch_size      = 25
BIGG_BATCH      = 27000
num_BIGG_BATCH  = 1
num_batch       = int((BIGG_BATCH-1000)/batch_size)
n=51
k               = 9
N1              = 2*k*102
N2              = n**2
semi1           = int(np.ceil(N1/8))*128
semi2           = int(np.ceil(N2/4))*128
latent_space    = 25                     ;   Z = 500    ; sigma1 = (0.01/np.sqrt(5)) ; sigma2 =0.0025
lr1             = 0.0001
lr2             = 0.0000000001


#########################################################
# Import Data
#########################################################
path  = Path('/home/cvl/Pycharm/pythonProject/Github')


Density_train = np.load(path / ('Density_Train.npy'), allow_pickle=True) +1
Density_test  = np.load(path / ('Density_Test.npy'), allow_pickle=True) +1
Density_train = np.expand_dims(Density_train, axis=2).astype('float32')
Density_test  = np.expand_dims(Density_test, axis=2).astype('float32')


Appr_Rho       = np.load(path / ('Linear_to_Nonlinear_Density.npy'), allow_pickle=True) +1
Appr_Rho_test  = np.expand_dims(Appr_Rho[27000:28000],axis=2)
Appr_Rho_train = np.expand_dims(Appr_Rho[0:27000],axis=2)


Surface_train = np.load(path / ('Surface_Train.npy'), allow_pickle=True)
Surface_test  = np.load(path / ('Surface_Test.npy'), allow_pickle=True)
Surface_train = np.reshape(Surface_train, [BIGG_BATCH, 2*n * k, 1])
Surface_train = np.concatenate((np.real(Surface_train), np.imag(Surface_train)), axis=1)
Surface_test  = np.reshape(Surface_test, [1000, 2*n * k, 1])
Surface_test  = np.concatenate((np.real(Surface_test), np.imag(Surface_test)), axis=1)


#########################################################
# Define Some Functions
#########################################################
def conv(input, w, strides):
    y = tf.nn.conv1d(input=input, filters=w, stride=strides, padding='SAME')
    y = tf.nn.leaky_relu(y)
    return y


def deconv(input, w, strides, output):
    y = tf.nn.conv1d_transpose(input=input, filters=w, strides=strides, padding='SAME', output_shape=output)
    y = tf.nn.leaky_relu(y)
    return y


def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

#########################################################
# Define Weights
#########################################################
def get_tfVariable(shape, name, trainable=True):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=50,)(shape), name=name, trainable=trainable, dtype=tf.float32)


############################ Encoder Density Weights
weights=[]
weights = weights + [get_tfVariable([3,1,32],   'W0')]
weights = weights + [get_tfVariable([3,32,64],  'W1')]
weights = weights + [get_tfVariable([3,64,128],  'W3')]

weights = weights + [get_tfVariable([semi2,Z],'W5')]
weights = weights + [get_tfVariable([Z],  'W12')]

weights = weights + [get_tfVariable([Z,latent_space],'W5')]
weights = weights + [get_tfVariable([latent_space],  'W12')]
weights = weights + [get_tfVariable([Z,latent_space], 'W6')]
weights = weights + [get_tfVariable([latent_space],  'W13')]

############################ Decoder Density Weights
weights = weights + [get_tfVariable([latent_space,Z], 'W7')]
weights = weights + [get_tfVariable([Z],    'W11')]

weights = weights + [get_tfVariable([Z,semi2], 'W7')]
weights = weights + [get_tfVariable([semi2],    'W11')]

weights = weights + [get_tfVariable([3,64,128], 'W9')]
weights = weights + [get_tfVariable([3,32,64], 'W11')]
weights = weights + [get_tfVariable([3,1,32],  'W12')]

############################ Decoder Surface Weights
weights = weights + [get_tfVariable([n*n,semi1], 'W7')]
weights = weights + [get_tfVariable([semi1],    'W11')]
weights = weights + [get_tfVariable([3,64,128], 'W9')]
weights = weights + [get_tfVariable([3,32,64], 'W11')]
weights = weights + [get_tfVariable([3,1,32],  'W12')]



########################################################
# Define Model
########################################################
def Latent_Sample(L):
    eps     = np.random.normal(0,1,[latent_space]).astype('float32')
    sample  = L[0] + tf.math.exp(L[1]/2)*eps
    return sample


def Encode(C):
     C = conv(C, weights[0], 2)
     C = conv(C, weights[1], 2)
     C = conv(C, weights[2], 1)
     C = tf.reshape(C, [C.shape[0], semi2])
     C = fullyConnected_layer(C, weights[3], weights[4])
     C = tf.nn.leaky_relu(C, alpha=1)

     mu  = fullyConnected_layer(C, weights[5], weights[6])
     var = fullyConnected_layer(C, weights[7], weights[8])
     return [mu,var]


def Decode( C, cond  ):
    C = fullyConnected_layer(C, weights[9], weights[10])
    C = tf.nn.leaky_relu(C, alpha=1)
    C = fullyConnected_layer(C, weights[11], weights[12])

    C  = tf.reshape(C, [C.shape[0], int(np.ceil(N2 / 4)), 128])
    C  = deconv(C, weights[13], 1, [C.shape[0], int(np.ceil(N2 / 4)), 64])
    C  = deconv(C, weights[14], 2, [C.shape[0], int(np.ceil(N2 / 2)), 32])
    C  = deconv(C, weights[15], 2, [C.shape[0], N2, 1])

    if  cond == 2 :
        C = C + tf.constant(np.random.normal(0, 1, [C.shape[0],N2,1]).astype('float32'))*sigma1#tf.math.exp(weights[16]/2)
        C = fullyConnected_layer(C[:,:,0], weights[16], weights[17])
        C = tf.nn.leaky_relu(C, alpha=1)
        C = tf.reshape(C, [C.shape[0], int(np.ceil(N1/8)), 128])
        C = deconv(C, weights[18],2,[C.shape[0],int(np.ceil(N1/4)),64])
        C = deconv(C, weights[19],2,[C.shape[0],int(np.ceil(N1/2)),32])
        C = deconv(C, weights[20],2,[C.shape[0],N1,1])

    return C

#########################################################
# Define Loss Function
#########################################################
def loss_function(L, y_pred, y_true, cond):

    Loss1 = tf.reduce_mean(tf.square(y_pred - y_true))
    Loss2 = tf.reduce_mean(-0.5 * tf.reduce_sum(L[1] - L[0] ** 2 - tf.math.exp(L[1]), axis=1))
    if cond==1 :
       Loss  = (sigma1**(-2))*Loss1 + 1*Loss2
    else :
        Loss  = (sigma2**(-2))*Loss1 + 1*Loss2
    return  [Loss1,Loss2,Loss]

#######################################################
#######################################################
def train_step(x_input, lr,  y_input = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        L      = Encode(x_input)
        Sample = Latent_Sample(L)
        if y_input is not None:
            preds  = Decode(Sample, cond=2)
            [Loss1, Loss2, Loss] = loss_function(L, preds , y_input ,cond=2)
        else :
            preds  = Decode(Sample, cond=1)
            [Loss1, Loss2, Loss] = loss_function(L, preds, x_input, cond=1)

        grads = tape.gradient(Loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return [Loss1, Loss2, Loss]


def Test_Score(epoch, x_input, y_input = None):
        L      = Encode(x_input)
        Sample = Latent_Sample(L)
        if y_input is not None:
            preds = Decode(Sample, cond=2)
            [Loss1, Loss2, Loss] = loss_function(L, preds , y_input ,cond=2)
        else :
            preds = Decode(Sample, cond=1)
            [Loss1, Loss2, Loss ] = loss_function(L, preds, x_input, cond=1)

        print("--- On epoch Test {} ---".format(epoch))
        tf.print(" ---Loss1:---", Loss1, " ---Loss2:---", Loss2, " ---Loss:---", Loss)

#######################################################
# Training Process
#######################################################

for EPOCH in range(train_epochs):

    if EPOCH >= 1:
        lr1 = 4 * lr1
        train_epochs1 = 8

    print('***********************************************************************')
    print("--- On  Main EPOCH {} ---  ".format(EPOCH))
    print('***********************************************************************')
    for epoch in range(train_epochs1):
        avg_Loss1 = 0 ; avg_Loss2 = 0 ; avg_Loss = 0
        if np.mod(epoch, 2) == 0:
            lr1 = lr1 / 2
        for j in range(num_batch):
            batch_x = Density_train[j * batch_size: (j + 1) * batch_size, :, :]
            [Loss1, Loss2, Loss] = train_step(batch_x, lr1)
            avg_Loss  += (Loss  / (num_batch))
            avg_Loss1 += (Loss1 / (num_batch))
            avg_Loss2 += (Loss2 / (num_batch))
        print("--- On DENSITY epoch {} --- ######################################### ".format(epoch))
        tf.print(" ---Loss1:---", avg_Loss1, " ---Loss2:---", avg_Loss2, " ---Loss:---", avg_Loss) ; print("\n")
        if (epoch  == train_epochs1-1):
            Test_Score(epoch,Density_test)



    lr2 = 1*lr1
    for epoch in range(train_epochs2):
        avg_Loss1 = 0 ; avg_Loss2 = 0 ; avg_Loss = 0
        if np.mod(epoch, 2) == 0:
            lr2 = lr2 / 2
        for j in range(num_batch):
             batch_x = Appr_Rho_train[j * batch_size: (j + 1) * batch_size, :, :]
             batch_y = Surface_train[ j*batch_size : (j+1)*batch_size , : , : ]
             [Loss1,Loss2,Loss]    = train_step(batch_x,lr2, y_input=batch_y)
             avg_Loss  +=  (Loss  / (num_batch))
             avg_Loss1 +=  (Loss1 / (num_batch))
             avg_Loss2 +=  (Loss2 / (num_batch))

        print("--- On SURFACE epoch {} --- @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ".format(epoch))
        tf.print(" ---Loss1:---", avg_Loss1 , " ---Loss2:---", avg_Loss2, " ---Loss:---", avg_Loss ) ; print("\n")
        if (epoch  == train_epochs2-1):
            Test_Score(epoch,Appr_Rho_test, y_input=Surface_test)

