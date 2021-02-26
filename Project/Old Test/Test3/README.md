# Test3
Permette la classificazione di immagini in 3 classi:<br/>
1. Muratura in stato normale<br/>
2. Muratura con delle rotture<br/>
3. Muratura con sfaldamento (Spalling)
<br/><br/>
Sono state spostate le funzioni principali in due file python JDUtils.py e JDModels.py, questo permette di evitare di avere codice duplicato (e magari con degli errori).<br/>
Se si utilizza google collab, conviene fare girare una volta il tutto con le variabili a True per l'aggiornamento delle librerie e il download del dataset. A fine di questo riavviare l'ambiente con le variabili a False. Queste sono situate nella cella dopo gli imports iniziali

## Dataset
Viene utilizzato il Dataset di default ed un dataset apposito per lo Spalling: <br/>
[Default Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2) <br/>
[Spalling Dataset](https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF/)

## Note
Per scrivere il notebook si è preso come riferimento iniziale: [Link](https://www.kaggle.com/vishnurapps/ensuring-building-safety-using-efficientnets) <br/>
Entrambi i notebook sono stati testati su Google Collab, utilizzando GPU, Si riesce ad ottenere un 95%+ di accuracy su validation e test set.
Ci sono ancora problematiche nel notebook "realuse" in cui si testa la rete su esempi reali (e su cui si esegue segmentazione)

## Citazioni
[IROS 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). Deep Concrete Inspection Using Unmanned Aerial Vehicle Towards CSSC Database. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), [Link PDF](https://ericlyang.github.io/img/IROS2017/IROS2017.pdf)<br/><br/>
[ROBIO 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). A Robotic System Towards Concrete Structure Spalling And Crack Database. 2017 IEEE Int. Conf. on Robotics and Biomimetics (ROBIO 2017), [Link Project](https://ericlyang.github.io/project/deepinspection/).
