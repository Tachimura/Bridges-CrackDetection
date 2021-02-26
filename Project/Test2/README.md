# Test2
Permette la classificazione di immagini in 3 classi:<br/>
1. Muratura in stato normale<br/>
2. Muratura con delle rotture<br/>
3. Muratura con sfaldamento (Spalling)

## Dataset
Viene utilizzato il Dataset di default ed un dataset apposito per lo Spalling: <br/>
[Default Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2) <br/>
[Spalling Dataset](https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF/)

## Note
Per scrivere il notebook si è preso come riferimento iniziale: [Link](https://www.kaggle.com/vishnurapps/ensuring-building-safety-using-efficientnets) <br/>
Per mancanza di una gpu il test è stato effettuato su cpu, il dataset è stato suddiviso in un subtest di dimensioni ridotte per
evitare tempistiche di training troppo lunghe. <br/>
Questa suddivisione può essere modificata a mano (sono presenti 3 intervalli che possiamo modificare, nella quale rimuoviamo i dati)
è inoltre presente un flag che se impostato a falso non esegue la suddivisione del dataset.<br/>
(Si è fatto questo per poter testare il corretto funzionamento del codice, si userà una qualche collaborazione esterna per poter addestrare la rete sull'intero dataset su una gpu dedicata: Google collab, ecc...)

## Citazioni
[IROS 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). Deep Concrete Inspection Using Unmanned Aerial Vehicle Towards CSSC Database. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), [PDF](https://ericlyang.github.io/img/IROS2017/IROS2017.pdf)<br/><br/>
[ROBIO 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). A Robotic System Towards Concrete Structure Spalling And Crack Database. 2017 IEEE Int. Conf. on Robotics and Biomimetics (ROBIO 2017), [Project](https://ericlyang.github.io/project/deepinspection/).
