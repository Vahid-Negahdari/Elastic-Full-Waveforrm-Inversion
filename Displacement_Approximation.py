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


#########################################################
# Import Data
#########################################################
path  = Path('/home/cvl/Pycharm/pythonProject/Github')
path2 = Path('/home/cvl/Pycharm/pythonProject/Github')

Surface_test      = np.load(path / ('surface27.npy'), allow_pickle=True)
Surface_test      = np.concatenate((np.real(Surface_test), np.imag(Surface_test)), axis=1)
Disp_Test         = {'Real':np.load(path / ('Disp_Real27.npy'), allow_pickle=True) ,
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
    preds = model(Surface_test)
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


    weights = []
    weights = weights + [get_tfVariable([3, k, 32], 'W0')]
    weights = weights + [get_tfVariable([3, 32, 64], 'W1')]
    weights = weights + [get_tfVariable([3, 64, 128], 'W3')]
    weights = weights + [get_tfVariable([3, 128, 256], 'W3')]
    weights = weights + [get_tfVariable([semi, k * 1 * n ** 2], 'W5')]
    weights = weights + [get_tfVariable([k * 1 * n ** 2], 'W6')]

    lr      = 0.01
    ################################################################
    ################################################################
    for epoch in range(train_epochs):
          avg_Loss = 0
          if np.mod(epoch,2)==0 :
             lr=lr/2
          for j in range(num_BIGG_BATCH):
                Surface = np.load(path / ('surface' + str(j) + '.npy'), allow_pickle=True)
                Surface = np.concatenate((np.real(Surface), np.imag(Surface)), axis=1)
                Disp    = np.load(path / ('Disp_'+ name1 + str(j) + '.npy'), allow_pickle=True)[:,a*n**2:(a+1)*n**2,:]
                Disp    = np.reshape(Disp, [BIGG_BATCH , k*n**2])
                if epoch < train_epochs - 1:
                   for s in range(num_batch):
                      batch_u   = Surface[s * batch_size: (s + 1) * batch_size]
                      batch_uu  = Disp[s * batch_size  : (s + 1) * batch_size ]
                      Loss      = train_step(batch_u,batch_uu)
                      avg_Loss  +=  Loss / (num_batch*num_BIGG_BATCH)
                else:
                    preds = model(Surface).numpy()
                    avg_Loss = loss_function(preds, Disp)
                    preds = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
                    np.save(path2 / ('Appr_Disp'+ str(a) + name1 + str(j) + '.npy'),preds)


          print("--- On epoch {} ---".format(epoch)) ; tf.print(" Loss:", avg_Loss) ; print("\n")
          if (epoch % 3 == 0):
             Test_Score(epoch,Disp_Test[name1][:,a*n**2:(a+1)*n**2,:])

    preds =  model(Surface_test).numpy()
    preds = np.reshape(preds, [BIGG_BATCH, n ** 2, k])
    np.save(path2 / ('Appr_Disp'+ str(a) + name1 + str(27) + '.npy'), preds)




del Surface
del Disp

for j in range(num_BIGG_BATCH + 1 ):
    pathh1 = path2/Path('Appr_Disp0Real' + str(j) + '.npy')
    pathh2 = path2/Path('Appr_Disp1Real' + str(j) + '.npy')
    A = np.load(pathh1, allow_pickle=True).astype('float32')
    B = np.load(pathh2, allow_pickle=True).astype('float32')
    np.save(path2 / ('Appr_Disp_Real' + str(j) + '.npy'),  np.concatenate(  (A, B) , axis=1)     )

    pathh3 = path2/Path('Appr_Disp0Complex' + str(j) + '.npy')
    pathh4 = path2/Path('Appr_Disp1Complex' + str(j) + '.npy')
    A = np.load(pathh3, allow_pickle=True).astype('float32')
    B = np.load(pathh4, allow_pickle=True).astype('float32')
    np.save(path2 / ('Appr_Disp_Complex' + str(j) + '.npy'),  np.concatenate(  (A, B) , axis=1)     )

    pathh1.unlink() ; pathh2.unlink() ; pathh3.unlink() ; pathh4.unlink()