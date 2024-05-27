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
BIGG_BATCH      = 28000
num_BIGG_BATCH  = 1
num_batch       = int((BIGG_BATCH-1000)/batch_size)
n=51
k               = 9
K               = list(range(0, 9, int(9/k)))
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
path = Path('/home/cvl/Pycharm/pythonProject/Final/L10K40/Super/Comprise_Final')
#path = Path(r'C:\Users\Vahid\PycharmProjects\pythonProject\Fieldd')

Dens      = np.load(path / ('Density_Group0.npy'), allow_pickle=True)
Dens      = -np.expand_dims(Dens, axis=2).astype('float32')
Dens_Test = Dens[27000:28000].copy()
Dens      = Dens[0:27000]

Fake_Dens = np.load(path / ('Nonlinear_Fake_Density_Group.npy'), allow_pickle=True)
Fake_Dens = -Fake_Dens.astype('float32')
Fake_Dens_Test = Fake_Dens[27000:28000].copy()
Fake_Dens = Fake_Dens[0:27000]


Data = np.load(path / ('surface_Group0.npy'), allow_pickle=True)[:,:,K]
Data = np.reshape(Data, [BIGG_BATCH, 102 * k, 1])
Data = np.concatenate((np.real(Data), np.imag(Data)), axis=1)
Data_Test = Data[27000:28000].copy()
Data      = Data[0:27000]

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
            [Loss1, Loss2, Loss,Loss3] = loss_function(L, preds , y_input ,cond=2)
        else :
            preds  = Decode(Sample, cond=1)
            [Loss1, Loss2, Loss , Loss3] = loss_function(L, preds, x_input, cond=1)

        grads = tape.gradient(Loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return [Loss1, Loss2, Loss, Loss3]


def Test_Score(x_input, y_input = None):
        L      = Encode(x_input)
        Sample = Latent_Sample(L)
        if y_input is not None:
            preds = Decode(Sample, cond=2)
            [Loss1, Loss2, Loss,Loss3] = loss_function(L, preds , y_input ,cond=2)
        else :
            preds = Decode(Sample, cond=1)
            [Loss1, Loss2, Loss , Loss3] = loss_function(L, preds, x_input, cond=1)

        print("--- On Test epoch TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ")
        tf.print(" ---Loss:---", Loss, " ---Loss1:---", Loss1, " ---Loss2:---", Loss2, " ---Loss3:---", Loss3)
        return [Loss1, Loss2, Loss, Loss3]

########################################################################################################################
########################################################################################################################


for EPOCH in range(train_epochs):

    print(EPOCH)
    if EPOCH >= 1:
        lr1 = 4 * lr1
        train_epochs1 = 8
 #       lr2 = 5 * lr2

    print('************************************************************************************************************')
    for epoch in range(train_epochs1):
        avg_Loss1 = 0 ; avg_Loss2 = 0 ; avg_Loss = 0 ; avg_Loss3 = 0
        if np.mod(epoch, 2) == 0:
            lr1 = lr1 / 2


        for j in range(num_batch):
            batch_x = Dens[j * batch_size: (j + 1) * batch_size, :, :]
            [Loss1, Loss2, Loss,Loss3] = train_step(batch_x, lr1)
            avg_Loss  += (Loss  / (num_batch))
            avg_Loss1 += (Loss1 / (num_batch))
            avg_Loss2 += (Loss2 / (num_batch))
            avg_Loss3 += (Loss3 / (num_batch))

        print("--- On DENSITY epoch $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {} ---".format(epoch))
        tf.print(" ---Loss:---", avg_Loss, " ---Loss1:---", avg_Loss1, " ---Loss2:---", avg_Loss2, " ---Loss3:---", avg_Loss3)
        print("\n")
        if (epoch  == train_epochs1-1):
            Test_Score(Dens_Test, lr1)


########################################################################################################################
########################################################################################################################
########################################################################################################################


    lr2 = 1*lr1
    for epoch in range(train_epochs2):
        avg_Loss1 = 0 ; avg_Loss2 = 0 ; avg_Loss = 0 ; avg_Loss3 = 0
        if np.mod(epoch, 2) == 0:
            lr2 = lr2 / 2


        for j in range(num_batch):
             batch_x = Fake_Dens[j * batch_size: (j + 1) * batch_size, :, :]
             batch_y = Data[ j*batch_size : (j+1)*batch_size , : , : ]
             [Loss1,Loss2,Loss,Loss3]    = train_step(batch_x,lr2, y_input=batch_y)
             avg_Loss  +=  (Loss  / (num_batch))
             avg_Loss1 +=  (Loss1 / (num_batch))
             avg_Loss2 +=  (Loss2 / (num_batch))
             avg_Loss3 +=  (Loss3 / (num_batch))

        print("--- On SURFACE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ epoch {} ---".format(epoch))
        tf.print(" ---Loss:---", avg_Loss , " ---Loss1:---", avg_Loss1 , " ---Loss2:---", avg_Loss2, " ---Loss3:---", avg_Loss3 )
        print("\n")
        if (epoch  == train_epochs2-1):
            Test_Score(Fake_Dens_Test,lr2, y_input=Data_Test)

#SAVE_WEIGHTS()
########################################################################################################################
########################################################################################################################
########################################################################################################################
# Direct  = -np.load(path / ('Direct_Real_Density_Group.npy'), allow_pickle=True).reshape([BIGG_BATCH,n,n])[27000:28000]
# Direct1 = -np.load(path / ('Direct_Fake_Density_Group1.npy'), allow_pickle=True).reshape([BIGG_BATCH,n,n])[27000:28000]
# Direct2 = -np.load(path / ('Direct_Fake_Density_Group2.npy'), allow_pickle=True).reshape([BIGG_BATCH,n,n])[27000:28000]
# Direct3 = -np.load(path / ('Direct_Fake_Density_Group3.npy'), allow_pickle=True).reshape([BIGG_BATCH,n,n])[27000:28000]
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
# L = encode(Fake_Dens[27000:28000])
# sample = latent_sample(L)
# SAMPLE =  Resample(L,0,1)
# preds = decode(sample,1)
# PREDS = decode(SAMPLE,1)
# #PREDS =  preds + (np.random.normal(0,1,[1000,N2,1])*sigma2).astype('float32')
# preds = np.reshape(preds,[1000,n,n])
# PREDS = np.reshape(PREDS,[1000,n,n])
# DENS  = np.reshape(Dens[27000:28000] , [1000,n,n])
# FAKE  = np.reshape(Fake_Dens[27000:28000] , [1000,n,n])
#
# def plott():
#     for j in range(100):
#         Z = [DENS[j] , FAKE[j],preds[j], PREDS[j],Direct[j],Direct1[j],Direct2[j],Direct3[j]]
#         m1 = np.min(Z) ; m2 = np.max(Z)
#
#         fig = plt.figure(figsize=(18.75, 6))
#         grid = ImageGrid(fig, 111,
#                          nrows_ncols=(2, 4), axes_pad=[0.15, 0.15], share_all=True, cbar_location="right", direction='row',
#                          cbar_mode="single", cbar_size="7%", cbar_pad=0.2, )
#         for i,ax in enumerate(grid):
#             im = ax.imshow(Z[i], vmin=m1, vmax=m2)
#             ax.cax.colorbar(im)
#             if i<4 :
#                 ax.set_title('First Method'+str(i), fontstyle='italic')
#          #   ax.set_title('First Method', fontstyle='italic')
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#         plt.savefig(str(j) + '.png')
#         plt.show()
#         plt.close()
#
# plott()
#
#
#
#
#
# sample=np.random.normal(0,1,[100,latent_space]).astype('float32')
# #sample = sample +  np.random.normal(0,0.4,[100,latent_space]).astype('float32')
# Gen = decode(sample,1)
# Gen = np.reshape(Gen,[100,n,n])
#
# def plot():
#
#   for kk in range(100):
#         T = Gen[kk]
#         S = Gen[kk]
#
#         DATA1 = np.reshape(T, [n, n])
#         DATA2 = np.reshape(S, [n, n])
#         Z = [DATA1, DATA2]
#         fig, axes = plt.subplots(nrows=2, ncols=1)
#         i = 0
#         for ax in axes.flat:
#             im = ax.imshow(Z[i], extent=[0, 1, 0, 1], vmin=np.min([np.min(Z[0]), np.min(Z[1])]),
#                            vmax=np.max([np.max(Z[0]), np.max(Z[1])]));
#             i = i + 1
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
#         plt.savefig(str(kk) + '.png')
#         plt.show()
#         plt.close()
#
#
# #plot()