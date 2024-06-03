import numpy as np
import tensorflow as tf
import scipy
import scipy.integrate
import scipy.special
from   pathlib import Path
from   io import BytesIO
from   zipfile import ZipFile
from   urllib.request import urlopen
import time

#########################################################
# Define Hyperparameter
#########################################################

n = 51  ; N = 2*n-1
l = 20  ; h = (2 * l) / (N - 1)
xy = int((N - 1) / 2)   ;  m = 2*n-1   ;  mm = int((m - 1) / 2)
n_u  = 9
num_BIG_BATCH = 28
BIG_BATCH     = 1000
mu = 1e+4  ; lambdaa = 1e+4
K  = 40

#########################################################
# Import Density and divide it
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


Density = np.load(path / ('Density_Train.npy'), allow_pickle=True)
Density = np.concatenate ( (Density , np.load(path / ('Density_Test.npy'), allow_pickle=True)) ,axis=0 )
for i in range(num_BIG_BATCH):
    np.save(path / ('Density' + str(i) + '.npy'), Density[i*BIG_BATCH:(i+1)*BIG_BATCH], allow_pickle=True)

#########################################################
# Import Incident Wave
#########################################################
ud  = np.load(path/('ud.npy'), allow_pickle=True)

#########################################################################################################################
# Generate Green Functions
#########################################################################################################################

def Mat_G():
    #########################################################
    # Set Domain Node
    #########################################################

    dim1 = np.linspace(-l, l, N)
    dim2 = np.linspace(-l, l, N)
    X, Y = np.meshgrid(dim1, dim2)
    XX = np.reshape(X, [N * N])
    YY = np.reshape(Y, [N * N])
    XXX = np.zeros([N ** 2, N ** 2])
    YYY = np.zeros([N ** 2, N ** 2])

    for i in range(N ** 2):
        XXX[i, :] = XX[i] - XX
        YYY[i, :] = YY[i] - YY
    #########################################################################################################################
    # Generate Green Functions
    #########################################################################################################################

    oo  = np.zeros([2, 2], dtype='complex64')
    phi = np.zeros([N, N, 2, 2], dtype='complex64')

    GG = np.zeros([2, 2], dtype='complex64')
    kp = K / np.sqrt(2 * mu + lambdaa)
    ks = K / np.sqrt(mu)

    f0 = lambda y, x:  np.real(scipy.special.hankel1(0, ks*((x**2 + y**2)**0.5) ))
    g0 = lambda y, x:  np.imag(scipy.special.hankel1(0, ks*((x**2 + y**2)**0.5) ))




    f1 = lambda y, x: (ks*(np.real(scipy.special.hankel1(1, ks * ((x ** 2 + y ** 2) ** 0.5)))) / ((x ** 2 + y ** 2) ** 0.5)) - \
                      (kp*(np.real(scipy.special.hankel1(1, kp * ((x ** 2 + y ** 2) ** 0.5)))) / ((x ** 2 + y ** 2) ** 0.5))

    g1 = lambda y, x: (ks*(np.imag(scipy.special.hankel1(1, ks * ((x ** 2 + y ** 2) ** 0.5)))) / ((x ** 2 + y ** 2) ** 0.5)) - \
                      (kp*(np.imag(scipy.special.hankel1(1, kp * ((x ** 2 + y ** 2) ** 0.5)))) / ((x ** 2 + y ** 2) ** 0.5))




    f2 = lambda y, x: ((ks**2)*(np.real(scipy.special.hankel1(2, ks * ((x**2 + y**2) ** 0.5)))) * (x**2) / (x**2 + y**2))-\
                      ((kp**2)*(np.real(scipy.special.hankel1(2, kp * ((x**2 + y**2) ** 0.5)))) * (x**2) / (x**2 + y**2))

    g2 = lambda y, x: ((ks ** 2) * (np.imag(scipy.special.hankel1(2, ks * ((x**2 + y**2) ** 0.5)))) * (x**2) / (x**2 + y**2))-\
                      ((kp ** 2) * (np.imag(scipy.special.hankel1(2, kp * ((x**2 + y**2) ** 0.5)))) * (x**2) / (x**2 + y**2))

    GG[0, 0] = (4*1j  / (4*mu)) *          scipy.integrate.dblquad(f0, 0, h / 2, lambda x: 0, lambda x: h / 2)[0] + \
               (4*1j  / (4*mu)) * 1j *     scipy.integrate.dblquad(g0, 0, h / 2, lambda x: 0, lambda x: h / 2)[0] + \
               (-4*1j / (4*K * K)) * scipy.integrate.dblquad(f1, 0, h / 2, lambda x: 0, lambda x: h / 2)[0] + \
               (-4*1j / (4*K*K))*1j* scipy.integrate.dblquad(g1, 0, h / 2, lambda x: 0, lambda x: h / 2)[0] + \
               (4*1j  / (4*K*K))   * scipy.integrate.dblquad(f2, 0, h / 2, lambda x: 0, lambda x: h / 2)[0] + \
               (4*1j /  (4*K*K))*1j* scipy.integrate.dblquad(g2, 0, h / 2, lambda x: 0, lambda x: h / 2)[0]

    GG[1, 1] = GG[0, 0]
    oo[:, :] = GG


    kp = K / np.sqrt(2 * mu + lambdaa)
    ks = K / np.sqrt(mu)

    F1 = lambda x, y: (1j/(4*mu))*scipy.special.hankel1(0, K * ((x ** 2 + y ** 2) ** 0.5)) - \
                          (1j/(4*K*K))*((ks*(scipy.special.hankel1(1, ks * ((x ** 2 + y ** 2) ** 0.5))) / (x ** 2 + y ** 2) ** 0.5) - \
                          (kp*(scipy.special.hankel1(1, kp * ((x ** 2 + y ** 2) ** 0.5))) / (x ** 2 + y ** 2) ** 0.5))

    F2 = lambda x, y: (1j/(4*K*K))*(((ks**2)*(scipy.special.hankel1(2, ks*((x**2 + y**2)**0.5))) / (x**2 + y**2)) - \
                          ((kp ** 2) * (scipy.special.hankel1(2, kp * ((x ** 2 + y ** 2) ** 0.5))) / (x ** 2 + y ** 2)))


    g1 = (h ** 2) * F1(X, Y)
    g2 = (h ** 2) * F2(X, Y)

    phi[:, :, 0, 0] = g1 + g2 * (X ** 2)
    phi[:, :, 1, 1] = g1 + g2 * (Y ** 2)
    phi[:, :, 0, 1] = g2 * (X * Y)
    phi[:, :, 1, 0] = g2 * (X * Y)

    phi = np.nan_to_num(phi)


    phi[xy, xy, :, :] = oo[:, :]

    phi[:, :, :, :] = (K ** 2) * phi[:, :, :, :]

    G1 = phi[:, :, 0, 0] ; GG1 = np.rot90(G1, 2)
    G2 = phi[:, :, 0, 1] ; GG2 = np.rot90(G2, 2)
    G4 = phi[:, :, 1, 1] ; GG4 = np.rot90(G4, 2)




    Ones = np.ones([n, n])
    Pad = np.pad(Ones, (n - 1, n - 1))

    P = np.argwhere(Pad == 1)

    A = np.zeros([2 * (n ** 2), 2 * (n ** 2)], dtype='complex64')
    A1 = np.zeros([n ** 2, n ** 2], dtype='complex64')
    A2 = np.zeros([n ** 2, n ** 2], dtype='complex64')
    A4 = np.zeros([n ** 2, n ** 2], dtype='complex64')

    #########################################################

    D = np.zeros([n ** 2, 4], dtype=int)
    for i in range(n ** 2):
        D[i, 0] = P[i, 1] - mm
        D[i, 1] = mm + n - 1 - P[i, 1]
        D[i, 2] = P[i, 0] - mm
        D[i, 3] = mm + n - 1 - P[i, 0]

    for i in range(n ** 2):
        A1[i, :] = GG1[mm - D[i, 2]: mm + D[i, 3] + 1, mm - D[i, 0]: mm + D[i, 1] + 1].flatten()
        A2[i, :] = GG2[mm - D[i, 2]: mm + D[i, 3] + 1, mm - D[i, 0]: mm + D[i, 1] + 1].flatten()
        A4[i, :] = GG4[mm - D[i, 2]: mm + D[i, 3] + 1, mm - D[i, 0]: mm + D[i, 1] + 1].flatten()

    A[0:n ** 2, 0:n ** 2] = A1
    A[0:n ** 2, n ** 2:2 * (n ** 2)] = A2
    A[n ** 2:2 * (n ** 2), 0:n ** 2] = A2
    A[n ** 2:2 * (n ** 2), n ** 2:2 * (n ** 2)] = A4

    np.save(path / 'A.npy', A)
    np.save(path / 'Green_Matrix.npy', [G1, G2, G4])

    return A


A = Mat_G()




########################################################################################################################
# Create Displacement Field and save it
########################################################################################################################

for i in range(3,num_BIG_BATCH):
    print(i)
    if (np.mod(i, 10) == 7):
         time.sleep(10 * 60)

    Density      = 1-np.load(path / ('Density' + str(i) + '.npy'), allow_pickle=True)
    Displacement = np.zeros([BIG_BATCH, 2*(n**2), n_u], dtype='complex64')
    Surface      = np.zeros([BIG_BATCH, 2 * n, n_u], dtype='complex64')
    RhoU         = np.zeros_like(Displacement)

    for j in range(BIG_BATCH):
        M  = Density[j, :]
        MM = np.concatenate((M,M),axis=0)
        u  = tf.linalg.solve(A*MM - np.eye(2*n**2,dtype='complex64'), -ud).numpy()
        Surface[j, 0:n, : ]   = u[0:n, :]     ;     Surface[j, n:2*n, :] = u[(n**2):(n**2)+n, :]
        Displacement[j, :, :] = u
        RhoU[j, :, :]         = u*np.expand_dims(MM,axis=1)


    np.save(path / ('Disp_Real' + str(i))   , np.real(Displacement))
    np.save(path / ('Disp_Complex' + str(i)), np.imag(Displacement))
    np.save(path / ('Surface' + str(i)), Surface)
    np.save(path / ('RhoU_Real' + str(i))   , np.real(RhoU))
    np.save(path / ('RhoU_Complex' + str(i)), np.imag(RhoU))