# Elastic Full-Waveform-Inversion : Physics-Guided Data-Driven Methods

## :art: About the project
paper [[1]](#1).
## :key: Getting Started
Please follow the guidelines we've provided to use the code effectively.
### Requirements
Please see the 
[requirements.txt](https://github.com/Vahid-Negahdari/Inverse_Elastic_Scattering/blob/main/requirements.txt) 
documentation for library and hardware requirements.
### Installing
1. Clone the repository to your local machine:
``` 
git clone https://github.com/Vahid-Negahdari/Inverse_Elastic_Scattering.git
```

2. Change directory into the cloned repository:
``` 
cd All-In-One-Python-Projects/<Project_name>
```
### Executing program

1.To execute the First Method, which is **Direct Deep Learning Inversion**:
``` 
python3 First-Method.py
```
2.To apply techniques within the Second Method, initially run:
``` 
python3 First-Method.py
```    
* To utilize the **Least Square** technique, execute:
  ``` 
  python3 First-Method.py
  ```
* To utilize the **Linear-to-Nonlinear** technique, execute:
  ``` 
  python3 First-Method.py
  ```  
* To utilize the **Inverse Convolution** technique, execute:
  ``` 
  This code will be completed soon.
  ```  
3.To utilize the Third Method, the **New-VAE** Method,
you first need to execute the linear-to-nonlinear process and then follow up with:
```
python3 First-Method.py
```
To utilize the Third Method, the New-VAE Method,
you first need to execute the linear-to-nonlinear process and then follow up with:

## :books: References 
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

## :relaxed: Author  
Vahid Negahdari

Email:  <vahid_negahdari@outlook.com>

Any discussions and contribution are welcomed!
