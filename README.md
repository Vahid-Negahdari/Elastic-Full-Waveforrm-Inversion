# Elastic FWI : Physics-Guided Data-Driven Methods 

## :art: About the project
In this project, we present methods based on data-driven techniques for the time-harmonic Elastic
Full-Waveform Inversion problem. Our project consists of several methods, including pure data-driven
technique (First Method), the integrated application of deep learning and the physics underlying 
the problem (Second Method) which includes several independent techniques, and a probabilistic
deep learning technique (Third Method). The full paper for this source code can be found on [[1]](#1).\
The dataset needed for this project [[2]](#2) is automatically uploaded within the codes. However, it's
important to mention that for the second and third methods to be employed, a larger dataset
had to be generated. Due to limitations on uploading data, we have included the necessary code
for creating auxiliary datasets.
## :key: Getting Started
Please follow the guidelines we've provided to use the code effectively.
### Requirements
Please see the 
[requirements.txt](https://github.com/Vahid-Negahdari/Inverse_Elastic_Scattering/blob/main/requirements.txt) 
documentation for library and hardware requirements.
### Installing
1. Clone the repository to your local machine:
``` 
git clone https://github.com/Vahid-Negahdari/Elastic-Full-Waveforrm-Inversion.git
```

2. Change directory into the cloned repository:
``` 
cd Elastic-Full-Waveforrm-Inversion
```
### Executing program

1.To execute the First Method, which is **Direct Deep Learning Inversion**:
``` 
python3 First_Method_Direct_DL.py
```
2.To apply techniques within the Second Method, initially run:
``` 
python3 Create_Dataset.py
python3 Displacement_Approximation.py
```    
* To utilize the **Least Square** technique, execute:
  ``` 
  python3 Second_Method_Least_Square.py
  ```
* To utilize the **Linear-to-Nonlinear** technique, execute:
  ``` 
  python3 RhoU_Approximation.py
  python3 Second_Method_Linear_to_Nonlinear.py
  ```  
* To utilize the **Inverse Convolution** technique, execute:
  ``` 
  This code will be completed soon
  ```  
3.To utilize the Third Method, the **New-VAE** Method,
you first need to execute the linear-to-nonlinear process and then follow up with:
```
python3 Third_Method_New_VAE.py
```

## :books: References 
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.\
<a id="2">[2]</a> 
Negahdari, Vahid (2024), “Elastic Full-Waveform-Inversion”,
Mendeley Data, V1, doi: 10.17632/z2n2f23pxw.1 <https://data.mendeley.com/datasets/z2n2f23pxw/1>

## :relaxed: Author  
Vahid Negahdari

Email:  <vahid_negahdari@outlook.com>

Any discussions and contribution are welcomed!
