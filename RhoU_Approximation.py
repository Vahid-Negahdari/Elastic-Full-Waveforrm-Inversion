import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path



#########################################################
# Define Hyperparameter
#########################################################

train_epochs   = 18
batch_size     = 25
num_BIGG_BATCH = 27
BIGG_BATCH     = 1000
num_batch      = int(BIGG_BATCH/batch_size)
lr             = 0.01
n              = 51
k              = 3
semi           = int(np.ceil(2*(n**2)/16)*256)

#########################################################
# Import Data
#########################################################

path  = Path('/home/cvl/Pycharm/pythonProject/Final/L10K40')
path2 = Path('/home/cvl/Pycharm/pythonProject/Final/L10K40/L2')

Surface_test      = np.load(path / ('surface27.npy'), allow_pickle=True)
Surface_test      = np.concatenate((np.real(Surface_test), np.imag(Surface_test)), axis=1)
Disp_Test         = {'Real'   :np.load(path / ('Disp_Real27.npy'), allow_pickle=True) ,
                     'Complex':np.load(path / ('Disp_Complex27.npy'), allow_pickle=True) }

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
def train_step(u,uu):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = model(u)
        current_loss = loss_function(preds, uu)
        grads = tape.gradient(current_loss, weights )
        optimizer.apply_gradients(zip(grads, weights ))
        return current_loss

def Test_Score(epoch,Disp):
    preds = model(Disp_test)
    loss = loss_function(preds, np.reshape(Disp,[BIGG_BATCH,k*n**2]))
    print("--- On epoch Test {} ---".format(epoch)) ; tf.print(" Loss:",loss) ; print("\n")

#######################################################
# Training Process
#######################################################

for p in range(4):
    print(p)
    if   p==0:
        name1 = 'Real'    ; a = 0
    elif  p==1:
        name1 = 'Real'    ; a = 1
    elif  p==2:
        name1 = 'Complex' ; a = 0
    elif  p==3:
        name1 = 'Complex' ; a = 1


    for t in range(3):
        print(t)
        K = list(range(3*t, 3*(t+1), 1))

        weights=[]
        weights = weights + [get_tfVariable([3,3,32],   'W0')]
        weights = weights + [get_tfVariable([3,32,64],  'W1')]
        weights = weights + [get_tfVariable([3,64,128],  'W3')]
        weights = weights + [get_tfVariable([3,128,256],  'W3')]
        weights = weights + [get_tfVariable([semi,k*1*n**2],'W5')]
        weights = weights + [get_tfVariable([1*k*n**2],   'W6')]

        lr = 0.01
        ########################################################
        ################################################################
        for epoch in range(train_epochs):
              avg_Loss = 0
              if np.mod(epoch,2)==0 :
                 lr=lr/2
              for j in range(num_BIGG_BATCH):
                    Disp_appr = np.load(path2 / ('Appr_Disp' +name1 + str(j) + '.npy' ), allow_pickle=True).astype('float32')[:,:,K]
                    RhoU      = np.load(path / ('RhoU_' + name1 + str(j) + '.npy'), allow_pickle=True)[:,a*n**2:(a+1)*n**2,K]
                    RhoU   = np.reshape  (RhoU, [BIGG_BATCH , k*n**2])
                    if epoch < train_epochs - 1:
                        for s in range(num_batch):
                              batch_u  = Disp_appr[s * batch_size: (s + 1) * batch_size]
                              batch_uu = RhoU[s * batch_size  : (s + 1) * batch_size ]
                              Loss     = train_step(batch_u, batch_uu)
                              avg_Loss += Loss / (num_batch*num_BIGG_BATCH)
                    else:
                        preds = model(Disp_appr).numpy()
                        avg_Loss = loss_function(preds, RhoU)
                        preds = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
                        np.save(path2 / (str(t) + 'Appr_RhoU' + str(a) + name1 + str(j) + '.npy'), preds)

              print("--- On epoch {} ---".format(epoch)) ; tf.print(" Loss:", avg_Loss) ; print("\n")
              if (epoch % 3 == 0):
                 Test_Score(epoch, RhoU_test[name1][:, a * n ** 2:(a + 1) * n ** 2, K])


        preds   =  model(Disp_appr_test).numpy()
        preds   = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
        np.save(path2 / (str(t) +'Appr_RhoU' + str(a) + name1 + str(27) + '.npy'), preds)



del Fake_sp
del Surface
del Scatt

for j in range(num_BIGG_BATCH+1):

    A1 = np.load(path2 / ('Fake_sp_Real_H' + str(0)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    A2 = np.load(path2 / ('Fake_sp_Real_H' + str(1)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    A3 = np.load(path2 / ('Fake_sp_Real_H' + str(2)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    A  = np.concatenate((np.concatenate((A1, A2), axis=2), A3), axis=2)


    B1 = np.load(path2 / ('Fake_sp_Real_V' + str(0)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    B2 = np.load(path2 / ('Fake_sp_Real_V' + str(1)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    B3 = np.load(path2 / ('Fake_sp_Real_V' + str(2)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    B = np.concatenate((np.concatenate((B1, B2), axis=2), B3), axis=2)

    np.save(path2 / ('Fake_splash_Real' + str(j) + '.npy'),  np.concatenate(  (A, B) , axis=1)     )



    C1 = np.load(path2 / ('Fake_sp_Complex_H' + str(0)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    C2 = np.load(path2 / ('Fake_sp_Complex_H' + str(1)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    C3 = np.load(path2 / ('Fake_sp_Complex_H' + str(2)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    C = np.concatenate((np.concatenate((C1, C2), axis=2), C3), axis=2)


    D1 = np.load(path2 / ('Fake_sp_Complex_V' + str(0)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    D2 = np.load(path2 / ('Fake_sp_Complex_V' + str(1)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    D3 = np.load(path2 / ('Fake_sp_Complex_V' + str(2)+'_' + str(j) + '.npy'), allow_pickle=True).astype('float32')
    D = np.concatenate((np.concatenate((D1, D2), axis=2), D3), axis=2)


    np.save(path2 / ('Fake_splash_Complex' + str(j) + '.npy'),  np.concatenate(  (C, D) , axis=1)     )