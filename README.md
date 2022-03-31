# Papers inspired by this research experiment

# Importance-Driven Deep Learning System Testing -ICSE 2020

### About
This paper [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3628024.svg)](https://doi.org/10.5281/zenodo.3628024)
presents DeepImportance, a systematic testing methodology accompanied by an Importance-Driven (IDC)
test adequacy criterion for DL systems. Applying IDC enables to
establish a layer-wise functional understanding of the importance
of DL system components and use this information to guide the
generation of semantically-diverse test sets. Our empirical evalua-
tion on several DL systems, across multiple DL datasets and with
state-of-the-art adversarial generation techniques demonstrates the
usefulness and effectiveness of DeepImportance and its ability to
guide the engineering of more robust DL systems.

### Repository
This repository includes details about the artifact corresponding to implementation of DeepImportance.
Our implementation is publicly available in
[DeepImportance repository](https://github.com/DeepImportance/deepimportance_code_release).
This artifact allows reproducing the experimental results presented in the paper. Below we
describe how to reproduce results. Before going further, first, check
installation page (i.e. INSTALL.md).


# My Research & Experiment
<a href="https://user-images.githubusercontent.com/48269869/158604449-15bd479e-89f6-4b06-8b61-0232cef89342.JPG" target="_blank">
<img src="https://user-images.githubusercontent.com/48269869/158604449-15bd479e-89f6-4b06-8b61-0232cef89342.JPG" alt="IMAGE ALT TEXT HERE" width="480" height="300" border="10" />

### Using Data Augmenation technique example
<a href="https://user-images.githubusercontent.com/48269869/160962853-71e5f255-fac5-4bc6-8288-3fc0a12b59a0.JPG" target="_blank">
<img src="https://user-images.githubusercontent.com/48269869/160962853-71e5f255-fac5-4bc6-8288-3fc0a12b59a0.JPG" alt="IMAGE ALT TEXT HERE" width="480" height="300" border="10" />
  
### Experiment step
<br>
1. First train data & make base model. ( In my case, I use ResNet32 model, epoch = 120, momentum = 0.9 , batch_size = 100, base_lr = 0.1) <br>
2. Measurement of various coverage evaluation values at base model. <br>
3. Make a data using data augmentaion. (I use dataaugmetaion like, point noise, spot noise, brightness change, weather change, zoom, rotation technique) <br>
4. Retrain using augmentation data & make retrain model. <br>
5. Measurement of various coverage evaluation values at retrain model. <br>
6. Filtering data like Null and Measuring Accuracy and Pearson's Correlation Coefficient
7. Compare Measuring value and analysis <br>
