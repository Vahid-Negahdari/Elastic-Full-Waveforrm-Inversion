# Elastic Full-Waveform-Inversion : Physics-Guided Data-Driven Methods

## :art: About the project
paper [[1]](#1).
## :key: Getting Started
In order to use this code, following this instruction:
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

1.For run the first method, **Direct Deep Learning Inversion** :
``` 
python3 First-Method.py
```
2.For use techniques within second method, first
``` 
python3 First-Method.py
```    
* For use **Least Square** technique
  ``` 
  python3 First-Method.py
  ```
* For use **Linear-to-Nonlinear** technique
  ``` 
  python3 First-Method.py
  ```  
* For use **Inverse Convolution** technique
  ``` 
  this code soon completed
  ```  
3.For use third method, **New-VAE** technique, initially need run linear-to-nonlinear technique and then run :
```
python3 First-Method.py
```


## :books: References 
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

## :relaxed: Author  
Vahid Negahdari

Email:  <vahid_negahdari@outlook.com>

Any discussions and contribution are welcomed!
