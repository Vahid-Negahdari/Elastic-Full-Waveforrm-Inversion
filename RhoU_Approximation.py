import tensorflow as tf
import numpy as np
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
k              = 3
semi           = int(np.ceil(2*(n**2)/16)*256)

#########################################################
# Import Data
#########################################################
path  = Path('/home/cvl/Pycharm/pythonProject/Github')

Disp_appr_test = {'Real'   : np.load(path / ('Appr_Disp_Real27.npy'), allow_pickle=True) ,
                  'Complex': np.load(path / ('Appr_Disp_Complex27.npy'), allow_pickle=True) }


RhoU_test      = {'Real'   : np.load(path / ('RhoU_Real27.npy'), allow_pickle=True) ,
                  'Complex': np.load(path / ('RhoU_Complex27.npy'), allow_pickle=True) }

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

def Model( u ):
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
def train_step(u,uu,lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = Model(u)
        current_loss = loss_function(preds, uu)
        grads = tape.gradient(current_loss, weights )
        optimizer.apply_gradients(zip(grads, weights ))
        return current_loss

def Test_Score(epoch,Disp,rhou):
    preds = Model(Disp)
    loss = loss_function(preds, np.reshape(rhou,[BIGG_BATCH,k*n**2]))
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
                    Disp_appr = np.load(path / ('Appr_Disp_' +name1 + str(j) + '.npy' ), allow_pickle=True).astype('float32')[:,:,K]
                    RhoU      = np.load(path / ('RhoU_' + name1 + str(j) + '.npy'), allow_pickle=True)[:,a*n**2:(a+1)*n**2,K]
                    RhoU   = np.reshape  (RhoU, [BIGG_BATCH , k*n**2])
                    if epoch < train_epochs - 1:
                        for s in range(num_batch):
                              batch_u  = Disp_appr[s * batch_size: (s + 1) * batch_size]
                              batch_uu = RhoU[s * batch_size  : (s + 1) * batch_size ]
                              Loss     = train_step(batch_u, batch_uu,lr)
                              avg_Loss += Loss / (num_batch*num_BIGG_BATCH)
                    else:
                        preds     = Model(Disp_appr).numpy()
                        Loss      = loss_function(preds, RhoU)
                        avg_Loss += Loss / (num_BIGG_BATCH)
                        preds     = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
                        np.save(path / (str(t) + 'Appr_RhoU' + str(a) + name1 + str(j) + '.npy'), preds)

              print("--- On epoch {} ---".format(epoch)) ; tf.print(" Loss:", avg_Loss) ; print("\n")
              if (epoch % 3 == 0):
                 Test_Score(epoch,Disp_appr_test[name1][:,:,K], RhoU_test[name1][:, a * n ** 2:(a + 1) * n ** 2, K])


        preds = Model(Disp_appr_test[name1][:,:,K]).numpy()
        preds = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
        np.save(path / (str(t) +'Appr_RhoU' + str(a) + name1 + str(27) + '.npy'), preds)



####################################################################
# Concatenate Rhou and derive Rho (divide RhoU by displacement)
####################################################################

for j in range(num_BIGG_BATCH+1):
    A = np.zeros([1000,2*(n**2),9],'float32')
    B = np.zeros([1000,2*(n**2),9],'float32')
    Appr_Rho = np.zeros([BIGG_BATCH, n ,n, 36], dtype=np.float32)
    for t in range(3):

        pathh1 = path/Path(str(t) + 'Appr_RhoU0Real'  + str(j) + '.npy')
        pathh2 = path/Path(str(t) + 'Appr_RhoU1Real'  + str(j) + '.npy')
        A1     = np.load(pathh1, allow_pickle=True).astype('float32')
        A2     = np.load(pathh2, allow_pickle=True).astype('float32')
        A[:,:,3*t:3*(t+1)]     = np.concatenate((A1, A2), axis=1)


        pathh3 = path/Path(str(t) + 'Appr_RhoU0Complex'  + str(j) + '.npy')
        pathh4 = path/Path(str(t) + 'Appr_RhoU1Complex'  + str(j) + '.npy')
        B1     = np.load(pathh3, allow_pickle=True).astype('float32')
        B2     = np.load(pathh4, allow_pickle=True).astype('float32')
        B[:,:,3*t:3*(t+1)]     = np.concatenate((B1, B2), axis=1)

        pathh1.unlink() ; pathh2.unlink() ; pathh3.unlink() ; pathh4.unlink()

    C = np.load(path / ('Appr_Disp_Real'+ str(j) + '.npy'), allow_pickle=True).astype('float32')
    D = np.load(path / ('Appr_Disp_Complex' + str(j) + '.npy'), allow_pickle=True).astype('float32')

    Appr_Rho[:, :, :, 0:9]   = np.reshape((A / C)[:, 0:n ** 2, :],      [BIGG_BATCH, n, n, 9])
    Appr_Rho[:, :, :, 9:18]  = np.reshape((A / C)[:, n**2:2*n ** 2, :], [BIGG_BATCH, n, n, 9])
    Appr_Rho[:, :, :, 18:27] = np.reshape((B / D)[:, 0:n ** 2, :],      [BIGG_BATCH, n, n, 9])
    Appr_Rho[:, :, :, 27:36] = np.reshape((B / D)[:, n**2:2*n ** 2, :], [BIGG_BATCH, n, n, 9])

    np.save(path / ('Appr_Rho' + str(j) + '.npy'), Appr_Rho )
