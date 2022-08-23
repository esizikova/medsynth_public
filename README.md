# Improving Computed Tomography (CT) Reconstruction via 3D Shape Induction
This repository contains code to replicate experiments from the paper:

Improving Computed Tomography (CT) Reconstruction via 3D Shape Induction

by Elena Sizikova (New York University), \
Xu Cao (New York University), \
Ashia Lewis (The University of Alabama), \
Kenny Moise (Université Quisqueya), \
Megan Coffee (NYU Grossman School of Medicine).

*Note*: code is based on the following implementation of TB Classifier by (Duong el al. 2021: "Detection of tuberculosis from chest X-ray images: Boosting the performance with vision transformer and transfer learning"):
https://github.com/linhduongtuan/Tuberculosis_ChestXray_Classifier


1. Clone the repository and set up the Python environment using the provided requirements.txt file:
```
git clone https://github.com/esizikova/medsynth_public.git
source /ext3/env.sh
```

### Pre-Train Reconstruction Model

2. Replace --dataroot in train_patchGAN_singleView256.sh and train_patchGAN_singleView256_finetuned.sh to path of the LIDC dataset (follow instructions <a href="https://github.com/kylekma/X2CT">here</a>). In our case, it is in: /LIDC-HDF5-256


Train baseline model for 90 epochs:
```
bash train_patchGAN_singleView256.sh
```

Alternatively, use our pre-trained model. Download pre-trained models from:
https://drive.google.com/drive/folders/1e37Qs37pXYcOf8EF5X3KgYUQCYnSy3Fz?usp=sharing

and place it into the folder medsynth_public/X2CT/3DGAN

```
save_models/singleView_CTGAN/LIDC256/d2_singleview2500_256/singleView_CTGAN/LIDC256/d2_singleview2500/checkpoint/90
```
should contain
```
90_net_D.pth and 90_net_G.pth
```

### Finetune Using Shape Induction Model

3. Replace TBX_PATH, MODEL_DEST_PATH, MODEL_SOURCE_PATH by path to TBX dataset, destination of finetuned model, and destination of pre-trained model

run
```
bash train_patchGAN_singleView256_finetuned.sh
```

or place downloaded 9_net_D.pth and 9_net_G.pth into
```
projection_model_experiments/1_norm/singleView_CTGAN/LIDC256/d2_singleview2500/checkpoint/9/
```


### Evaluate Using Classification Experiments

4. Download TBX11K data TBX11K_classification_splits data and place into ROOT/data directory, e.g., /scratch/es5223/medsynth_clean/data/TBX11K_classification_splits
```
python generate_CT_for_TBX.py test/tb/
```

5. Generate synthetic CT images:
* Modify model_save_directory in generate_CT_for_TBX.py to point to the correct model.

* Modify DATA_SOURCE and OUT_SOURCE in generate_CT_for_TBX.py to point to source and target directories of the data, respectively.

* Run to generate synthetic CT data:
```
cd classification
python generate_CT_for_TBX.py test/tb
python generate_CT_for_TBX.py val/tb
python generate_CT_for_TBX.py train/tb
python generate_CT_for_TBX.py test/health
python generate_CT_for_TBX.py val/health
python generate_CT_for_TBX.py train/health
python generate_CT_for_TBX.py test/sick
python generate_CT_for_TBX.py val/sick
python generate_CT_for_TBX.py train/sick
```

6. Run classification training for CT only models:
```
python 3D_CNN_bce.py
python 3D_CNN_bce.py proj
```

7. Run classification training for X-ray only models:
```
python VT_X.py orig
```

8. Run classification training for X-ray and CT models:
```
python X+CT-joinedModel-3DCNN.py orig
python X+CT-joinedModel-3DCNN.py proj
```


9. Evaluate and compare model performance:
```
jupyter notebook evaluate_pretrained_models.ipynb
```

# BibTeX
```
@article{ShapeInductionSizikova22,
title = {Improving Computed Tomography (CT) Reconstruction via 3D Shape Induction},
year = {2022},
author = {E. Sizikova and X. Cao and A. Lewis and K. Moise and M. Coffee},
}
```


