# Ripa Layout Analysis and OCR


The repository host scripts and models for the layout analysis of Ripa's Edition.

The notebook RTK.ipynb use the [RTK pipeline](https://github.com/PonteIneptique/rtk) to combine YALTAi and Kraken.

For Layout Analysis it use [LADaS](https://github.com/DEFI-COLaF/LADaS/releases/tag/2024-10-31).

Kraken use [CATMuS-Print Large](https://zenodo.org/records/10592716)

A Big Thank you to [Simon Gabay](https://github.com/gabays) for the help! 

For transformation of XML Alto to TEI : https://github.com/DEFI-COLaF/LADAS2TEI

# Updates for training

Dataset: https://app.roboflow.com/acdic/ripa-bsyqg/1/export

"Download dataset"->yolo12

Point in train_model.py 
DATA_YAML_PATH = 'RIPA-ft/data.yaml' Yaml included in Roboflow download

Fine tuned model in my_finetune_project/run_ladas_1280_l_v14
/weights/

# Updates for eval:

eval.ipynb uses same data.yaml as training

# Updates for tei/altoxml

experimental non-RTK code in altoxml.ipynb
