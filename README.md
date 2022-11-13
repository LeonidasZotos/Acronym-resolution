# Acronym-resolution

## Goal
This system focuses on identifying, disambiguating and semamtically expanding acronyms in a given text.

## Repository Structure
The code used for training is independent from the code used to run the entire system. A trained model is needed to run the entire system.
1.  "src" contains the source code for the model training (*train.py*) and system (*main.py*, *model.py*, *semanticExpansion.py*, *acronymDisambiguator.py* & *utils.py*). The evaluation of the entire (disambiguation and semantic expansion) system can be done by running *main.py*. To only test the disambigution, *test.py* can be used.
2.  The "src/model" folder contains a configuration file as well as the two models. The two trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Ar1639i9Sg5gNP4Qfeod4yOYgyi2ujNJ?usp=share_link) and should be placed in "src/model"
3.  The "science" & "scienceMed" folders contain the data and vocabularies used to fine-tune the respective BERT models. 
4.  The "input" folder can contain *.txt* or *.csv* files. The system will expand all *.txt* and *.csv* files in that folder.
5.  The "output" folder contains the expanded files. Each file is named after its original filename. 
6.  "exampleTestFiles" contains some example *.csv* files that can be used to test either the "science" or "scienceMed" models.

## How to Run
A python (tested on v3.9) environment with the requirements listed in *requirements.txt* is needed for either training or running the system

### Training the models
1. From within *src/*, the following command can be used to train the model, where dataset is either "science" or "scienceMed" depending on the dataset on which the model should be fine-tuned (default is "science").
   ```
   python train.py [dataset] 
   # example: "python train.py science"
   ```
2. The produced model is created in the *src/* and will be named "model.bin". If this model is to be used, it should be moved to "src/model" and its name should change to either "scienceMedModel.bin" or "scienceModel.bin", depending on which dataset it was trained on.

### Running the detection, disambiguation & Expansion system
For the system to work, a fine-tuned *.bin* model needs to be present in "src/model".
1. From within *src/*,  the following command can be used to expand the text files locaetd in *input/*. *inputFolderLocation* is the location of the *input* folder, while *modelType* is the type of model to be used (either "science" or "scienceMed")
   ```
   python main.py [inputFolderLocation] [modelType] # example: 
   "python train.py ../input scienceMed"
   ```
2. The output is produced in *output/*. A summary of the evaluation for the disambiguation & semantic expansion is produced within the Command-Line Interface once the execution is complete. 

## Resources
The acronym disambiguator code is based on this [Huggingface space](https://huggingface.co/spaces/kpriyanshu256/acronym-disambiguation). 

### SDU @ AAAAI-21 Dataset
Scientific Acronym training examples and Lexicon, to cite:
```
@inproceedings{veyseh-et-al-2020-what,
   title={{What Does This Acronym Mean? Introducing a New Dataset for Acronym Identification and Disambiguation}},
   author={Amir Pouran Ben Veyseh and Franck Dernoncourt and Quan Hung Tran and Thien Huu Nguyen},
   year={2020},
   booktitle={Proceedings of COLING},
   link={https://arxiv.org/pdf/2010.14678v1.pdf}
}
```

### UMN Dataset
Medical Acronym Lexicon, to cite:
```
@article{moon2012clinical,
  title={Clinical Abbreviation Sense Inventory},
  author={Moon, Sungrim and Pakhomov, Serguei and Melton, Genevieve},
  year={2012}
}
```
