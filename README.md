# tesi-rn-ponti
Progetto tesi riconoscimento crepe nei ponti attraverso image recognition

## Setup

|           | Version |
|-----------|---------|
| Python    | 3.7     |

## Environment
Nella cartella Environment sarà presente il file environment.yml per creare un environment contenente già tutti gli imports necessari per il funzionamento del progetto, possibili aggiornamento di questo saranno specificati nel documento README della cartella.

## Dataset
Il dataset utilizzato è presente nella cartella Dataset e potrà ricevere modifiche nei momenti successivi di sviluppo.<br/>
Il dataset di partenza scelto è:<br/>
[Default Dataset](https://www.kaggle.com/arunrk7/surface-crack-detection)

### Python Environment

* Install conda/anaconda distribution

* Create a new conda environment:

  > `conda env create -f environment.yml`
  >
  > `conda activate rn_ponti`

* Test if all dependencies are satisfied (make you sure to be in the repository directory,all commands in this guide are relative to the repository):
  > `cd Environment`

  > `python test_dependencies.py`

  If *"All dependencies are satisfied"* message show up, everything has been setup correctly!.

  Remember to **activate** rn_ponti conda environment before to write any code! To deactivate:
  > `conda deactivate`

  To delete the environment (only the environment is deleted, the project directory is a separate entity!):
  > `conda deactivate`
  >
  > `conda env remove --name rn_ponti`
