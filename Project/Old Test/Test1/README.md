# Test1
Permette la classificazione binaria di 'Muratura in stato normale' e 'Muratura con delle rotture'

## Dataset
Viene utilizzato il Dataset di default: <br/>
[Default Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## Note
Per scrivere il notebook si è preso come riferimento iniziale: [Link](https://www.kaggle.com/vishnurapps/ensuring-building-safety-using-efficientnets) <br/>
Per mancanza di una gpu il test è stato effettuato su cpu, il dataset è stato suddiviso in un subtest di dimensioni ridotte per
evitare tempistiche di training troppo lunghe. <br/>
Questa suddivisione può essere modificata a mano (sono presenti 3 intervalli che possiamo modificare, nella quale rimuoviamo i dati)
è inoltre presente un flag che se impostato a falso non esegue la suddivisione del dataset.<br/>
(Si è fatto questo per poter testare il corretto funzionamento del codice, si userà una qualche collaborazione esterna per poter addestrare la rete sull'intero dataset su una gpu dedicata: Google collab, ecc...)
