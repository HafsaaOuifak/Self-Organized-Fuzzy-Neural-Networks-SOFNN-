### Installation 

To install requirements in windows cmd:
FOR /F %k in (requirements.txt) DO ( if NOT # == %k ( pip install %k ) )

If you have any problems try instlling the version 1 of tensorflow in an independent conda environment, then install
manually the other needed packages.


### For training 
python main.py --dataset wbc

### Work with trained models
python main_trained.py --dataset wbc

### Datasets 
Change "wbc" with other datasets from the list or add your own.

wisconsin: wbc
Diabets: diabets
Parkinsons: parkinson
SPECTF: spectf
