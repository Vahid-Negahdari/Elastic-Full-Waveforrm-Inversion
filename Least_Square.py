import tensorflow as tf
import numpy as np
from pathlib import Path


path = Path('/home/cvl/Pycharm/pythonProject/Github')

BIGG_BATCH  = 500
n  = 51
k  = 9


A   = np.load(path / ('A.npy'), allow_pickle=True)
ud  = np.load(path / ('ud.npy'), allow_pickle=True)
Rho = np.load(path / ('Density' + str(27) + '.npy'), allow_pickle=True)
U1  = np.load(path / ('Appr_Disp_Real' + str(27) + '.npy'), allow_pickle=True)
U2  = np.load(path / ('Appr_Disp_Complex' + str(27) + '.npy'), allow_pickle=True)
U   = U1 + 1j*U2





Appr_Rho = np.zeros([BIGG_BATCH,n**2])

for i in range(BIGG_BATCH):
    print(i)
    C  = np.zeros([k*4*n**2 , n**2])
    Y  = np.zeros([k*4*n**2 , 1])
    for j in range (k):
        B = A*U[i,:,j]
        B = B[:, 0:n**2] + B[:, n**2 : 2*n**2]
        B = np.concatenate(  (np.real(B) , np.imag(B)) , axis=0)
        C [ j*4*n**2  : (j+1)*4*n**2 , : ]   = B.copy()
        y = U[i,:,j] - ud[:,j]
        y = np.concatenate(  (np.real(y) , np.imag(y)) , axis=0)
        Y [ j*4*n**2: (j+1)*4*n**2, 0] = y.copy()
    Appr_Rho[i] = 1-tf.linalg.lstsq(C,Y,l2_regularizer=0.02).numpy()[:,0]


print( np.mean( np.sqrt( np.sum(np.square(Appr_Rho-Rho[0:500]),axis=1)/  np.sum(np.square(Rho[0:500]),axis=1)  ) ))