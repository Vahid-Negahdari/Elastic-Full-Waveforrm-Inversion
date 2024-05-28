import tensorflow as tf
import numpy as np
from numpy import inf
from pathlib import Path


#########################################################
# Define Hyperparameter
#########################################################
train_epochs    = 20
batch_size      = 25
BIGG_BATCH      = 1000
num_BIGG_BATCH  = 27
num_batch       = int(BIGG_BATCH/batch_size)
n               = 51
latent_space    = 1000
semi            = int(np.ceil(n/4))*int(np.ceil(n/4))*128
lr              = 0.003

#########################################################
# Import Data
#########################################################

path  = Path('/home/cvl/Pycharm/pythonProject/Github')

Density_test  = np.load(path / ('Density27.npy'), allow_pickle=True)
Density_test  = np.reshape(Density_test, [BIGG_BATCH, n, n, 1])

Appr_Rho_test = np.load(path / ('Appr_Rho27.npy'), allow_pickle=True)
Appr_Rho_test[np.abs(Appr_Rho_test) == inf] = 0
Appr_Rho_test[np.abs(Appr_Rho_test) > 12] = np.mean(Appr_Rho_test)

#########################################################
# Define Some Functions
#########################################################
def conv(input, w, strides):
    y = tf.nn.conv2d(input=input, filters=w, strides=strides, padding='SAME')
    y = tf.nn.leaky_relu(y)
    return y

def deconv(input, w, strides, output):
    y = tf.nn.conv2d_transpose(input=input, filters=w, strides=strides, padding='SAME', output_shape=output)
    y = tf.nn.leaky_relu(y)
    return y


def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

#########################################################
# Define Weights
#########################################################
def get_tfVariable(shape, name):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=50,)(shape), name=name, trainable=True, dtype=tf.float32)

weights=[]
weights = weights + [get_tfVariable([3,3,36,64],  'W1')]
weights = weights + [get_tfVariable([3,3,64,64],  'W2')]
weights = weights + [get_tfVariable([3,3,64,128],  'W3')]


weights = weights + [get_tfVariable([semi,latent_space],     'W5')]
weights = weights + [get_tfVariable([latent_space],          'W12')]

weights = weights + [get_tfVariable([latent_space,semi],    'W7')]
weights = weights + [get_tfVariable([semi],  'W11')]


weights = weights + [get_tfVariable([3,3,64,128], 'W9')]
weights = weights + [get_tfVariable([3,3,64,64], 'W10')]
weights = weights + [get_tfVariable([3,3,36,64],  'W11')]
weights = weights + [get_tfVariable([3,3,1,36],  'W11')]

########################################################
# Define Model
########################################################
def Model( input ):
    C1 = conv(input, weights[0],2)
    C2 = conv(C1, weights[1],2)
    C3 = conv(C2, weights[2],1)

    C4 = tf.reshape(C3,[C3.shape[0],semi])
    C5 = fullyConnected_layer(C4, weights[3], weights[4])

    C = fullyConnected_layer(C5, weights[5], weights[6]) + C4
    C = tf.reshape(C, [C.shape[0], 13,13,128])           + C3

    C = deconv(C, weights[7],1,[C.shape[0],13,13,64])   + C2
    C = deconv(C, weights[8],2,[C.shape[0],26,26,64])   + C1
    C = deconv(C, weights[9],2,[C.shape[0],51,51,36])  + input
    C = deconv(C, weights[10],1,[C.shape[0],51,51,1])

    return C

#########################################################
# Define Loss Function
#########################################################
def loss_function(y_pred, y_true):
    Loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return  Loss

#######################################################
#######################################################
def train_step(x_input,y_input,lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds       = Model(x_input)
        Loss = loss_function(preds,y_input)
        grads = tape.gradient(Loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return Loss



def Test_Score(epoch):
    preds = Model(Appr_Rho_test)
    loss = loss_function(preds, Density_test)
    print("--- On epoch Test {} ---".format(epoch)) ; tf.print(" Loss:",loss) ; print("\n")


#######################################################
# Training Process
#######################################################
Appr_Final = np.zeros([28000,n**2],'float32')

for epoch in range(train_epochs):
    avg_Loss = 0
    if np.mod(epoch, 2) == 0:
        lr = lr / 2
    for i in range(int(num_BIGG_BATCH)):
         Density = np.load(path / ('Density' + str(i)+'.npy'), allow_pickle=True)
         Density = np.reshape(Density,[BIGG_BATCH,n,n,1])

         Appr_Rho = np.load(path / ('Appr_Rho' + str(i) + '.npy'), allow_pickle=True)
         Appr_Rho[np.abs(Appr_Rho) == inf] = 0
         Appr_Rho[np.abs(Appr_Rho) > 12] = np.mean(Appr_Rho)

         if epoch < train_epochs - 1:
             for j in range(int(num_batch)):
                 batch_y   = Density[ j*batch_size : (j+1)*batch_size ]
                 batch_x   = Appr_Rho  [ j*batch_size : (j+1)*batch_size ]
                 Loss      = train_step(batch_x,batch_y,lr)
                 avg_Loss +=  (Loss / (num_batch*num_BIGG_BATCH))
         else :
             preds     = Model(Appr_Rho).numpy()
             Loss      = loss_function(preds, Density)
             avg_Loss += (Loss / (num_BIGG_BATCH))
             Appr_Final[i*BIGG_BATCH:(i+1)*BIGG_BATCH] = np.reshape(preds, [BIGG_BATCH, n ** 2])
    print("--- On epoch {} ---".format(epoch)) ; tf.print(" Loss:", avg_Loss) ; print("\n")
    if (epoch % 3 == 0):
       Test_Score(epoch)

preds                   = Model(Appr_Rho_test).numpy()
Appr_Final[27000:28000] = np.reshape(preds, [BIGG_BATCH, n ** 2])
Density_test  = 1-np.load(path / ('Density27.npy'), allow_pickle=True)

d1 = np.sqrt(np.sum(np.square(Appr_Final[27000:28000]-Density_test),axis=1)) / np.sqrt(np.sum(np.square(1-Density_test),axis=1))
print(np.mean(d1))

np.save(path / ('Linear_to_Nonlinear_Density.npy'), Appr_Final)

