## HALOS: Hallucination-free Organ Segmentation after Organ Resection Surgery


[![Conference Paper](https://img.shields.io/static/v1?label=DOI&message=10.1007%2f978-3-030-87240-3_66&color=3a7ebb)](https://doi.org/10.1007/978-3-031-34048-2_51)
[![Preprint](https://img.shields.io/badge/arXiv-2107.05990-b31b1b)](https://arxiv.org/abs/2303.07717)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code of the paper: "HALOS: Hallucination-free Organ Segmentation after Organ Resection Surgery"
Authors: [Murong Xu, murong-xu](https://github.com/murong-xu), [Anne-Marie Rickmann, arickm](https://github.com/arickm)

If you're using our code, please cite:
```
@inproceedings{rickmann2023halos,
  title={HALOS: Hallucination-Free Organ Segmentation After Organ Resection Surgery},
  author={Rickmann, Anne-Marie and Xu, Murong and Wolf, Tom Nuno and Kovalenko, Oksana and Wachinger, Christian},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={667--678},
  year={2023},
  organization={Springer}
}
```

This repository contains original code of the DAFT module, introduced in the paper "DAFT: A Universal Module to Interweave Tabular Data and 3D Images in CNNs"
the original DAFT repostitory can be found at: https://github.com/ai-med/DAFT

If you're using HALOS code, please also cite:
```
@article{Wolf2022-daft,
  title = {{DAFT: A Universal Module to Interweave Tabular Data and 3D Images in CNNs}},
  author = {Wolf, Tom Nuno and P{\"{o}}lsterl, Sebastian and Wachinger, Christian},
  journal = {NeuroImage},
  pages = {119505},
  year = {2022},
  issn = {1053-8119},
  doi = {10.1016/j.neuroimage.2022.119505},
  url = {https://www.sciencedirect.com/science/article/pii/S1053811922006218},
}
```
and the paper "Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform"

```
@inproceedings(Poelsterl2021-daft,
  title     = {{Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform}},
  author    = {P{\"{o}}lsterl, Sebastian and Wolf, Tom Nuno and Wachinger, Christian},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  pages     = {688--698},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.05990},
  doi       = {10.1007/978-3-030-87240-3_66},
}
```

Further, the implementation of the backbone U-Net in HALOS is based on the old version of nnU-Net: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)

please also cite:

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
for deep learning-based biomedical image segmentation. Nature Methods, 1-9.


## Docker:
A docker image to run our code can be found here: https://github.com/orgs/ai-med/packages/container/package/halos


## data structure:

HALOS uses image data and binary labels indicating if the patient had a cholecystectomy or nephrectomy. We define 0 as no surgery and 1 as post- organ resection surgery.

In config.py you need to set the paths for your data.

The data should be structured like this:

data
  - gallbladder
    - segmentation
        - dict_test.pickle : python dictionary with keys ‘0’ and ‘1’ containing lists of subject ids belonging to the corresponding label. e.g. {0: ['0100013'], 1: ['1028818']} for the test set
        - binary_label_all.pickle  = python dictionary with keys ‘0’ and ‘1’ containing lists of subject ids belonging to the corresponding label. e.g. {0: ['0100013'], 1: ['1028818']}
        - data
          - <subject_id>
            - mri_opp.nii.gz : MRI scan
            - annotation.nii.gz : segmentation
        - seg_folds : contains txt files with subject id’s for each fold
          - fold1_train.txt
          - fold1_val.txt
          - fold1_test.txt
    - UKB
      - train_binary_labels.pickle : binary labels for training set
      - test_binary_labels.pickle : binary labels for test set
      - test
        - <subject_id>
            - mri_opp.nii.gz : MRI scan
      - train
        - <subject_id>
           - mri_opp.nii.gz : MRI scan
  - kidney (same setup as for gallbladder)


## training:

to train a HALOS model call:

```
python training/train.py
```

you can also pass hyperparameters e.g. the hyperparameters used in the paper:

```
python training/train.py --fusion_input gt --fusion_loca 2 --clf_feature_loca 5 --loss_weight_seg 0.7 --seg_lr 0.000818 --clf_lr 0.00087 --weight_decay 0.000061 --val_interval 4 --base_num_features 32 --clf_model CLFCNN --clf_update ukb_only --norm BN  --clf_factor 8 --fusion_squeeze_factor 4
```

hyperparameters can also be adapted in config/config.py


## logging:

our scripts are using [wandb](https://wandb.ai/site) for logging training and evaluation


## evaluation

for testing there are 2 different scripts to evaluate the segmentation data and UKB data in eval:

you just have to adapt the paths to your saved model in the scripts.

```
python eval_ukb.py
```

```
eval_segmentation.py
```



