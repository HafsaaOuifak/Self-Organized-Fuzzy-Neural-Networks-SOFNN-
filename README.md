# Self-Organizing Fuzzy Neural Network (SOFNN)
This version allows training a SOFNN, and counting rules and membership functions.

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

### Publication

The results have been evaluated and presented in our paper:

"On the performance and interpretability of Mamdani and Takagi-Sugeno-Kang based neuro-fuzzy systems for medical diagnosis"

You can cite it as:

Hafsaa Ouifak and Ali Idri. On the performance and interpretability of mamdani and takagi-sugeno-kang based neuro-fuzzy systems for medical diagnosis. Scientific African, 20:e01610, 2023, doi:10.1016/j.sciaf.2023.e01610.
