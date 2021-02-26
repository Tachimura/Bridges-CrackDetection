# tesi-rn-ponti
Progetto tesi riconoscimento crepe nei ponti attraverso image recognition

## Setup

|           | Version |
|-----------|---------|
| Python    | 3.7     |

## Environment
Nella cartella Environment sarà presente il file environment.yml per creare un environment contenente già tutti gli imports necessari per il funzionamento del progetto.<br/>
Possibili aggiornamenti a questo verranno definiti nel documento README della cartella.

## Dataset
Il dataset utilizzato è presente nella cartella Dataset e potrà ricevere modifiche nei momenti successivi di sviluppo.<br/>
Il dataset di partenza scelto è:<br/>
[Default Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

### Python Environment

* Install conda/anaconda distribution

* Create a new conda environment:

  > `Modify the last row of environment.yml so it links the environment into your 'envs' pc account directory`
  >
  > `conda env create -f Environment/environment.yml`
  >
  > `conda activate rn_ponti`

* Test if all dependencies are satisfied (make you sure to be in the repository directory,all commands in this guide are relative to the repository):
  > `cd Environment`

  > `python test_dependencies.py`

  If *"All dependencies are satisfied"* message show up, everything has been setup correctly!.

  Remember to **activate** rn_ponti conda environment before writing any code!<br/>
 * To deactivate:
  > `conda deactivate`

  To delete the environment (only the environment is deleted, the project directory is a separate entity!):
  > `conda deactivate`
  >
  > `conda env remove --name rn_ponti`
