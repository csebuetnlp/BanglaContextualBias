# BanglaBertBiasDetection
A repository for the bias detection in Bangla Language Model

## Requirements
All the requirements are given in the `requirements.txt` file. To install the requirements, run the following command:
```console
./installation.sh
```

## Data
### Category Words
The words for WEAT categories are given in `CeatDataCollection\CEAT_Categories` folder. 
### CEAT Data
The full dataset is not uploaded due to the size of the dataset. CEAT datasets that were used are in raw unprocessed format. A sample dataset is given in `CeatDataCollection\sample_data` folder. 
### Log Probability Data
The experiments were conducted taking positive and negative traits from `data\positive_traits_bn.csv` and `data\negative_traits_bn.csv` respectively. 

## Usage
### WEAT Test
Details for WEAT test are given in `Notebooks/Exposing_Bias_by_WEAT_Test.ipynb` file.
### CEAT : Sentence Extraction
For extracting sentences from a single file, run the following command:
```
python ./CEATDataCollection/extractSentences.py -f <path_to_file>
```
For extracting sentences for multiple files from multiple directories, run the following command:
```
python ./CEATDataCollection/extractSentences.py -dir <path_to_dir1> <path_to_dir2> ... <path_to_dirN>
```
After extracting the sentences, all the sentences are combined using the following command:
```
python ./CEATDataCollection/combineResults.py
```
The combined results are stored in `CEATDataCollection/results/results_trait.pkl` file. During combining, all the data are collected from folders where `sentences.csv` file exists.
### CEAT : Embedding Extraction
For extracting embeddings from a , run the following command:
```
python ./CEATDataCollection/extractEmbeddings.py -l <segment_length>
```
This will take data from `CEATDataCollection/results/results_trait.pkl` file and extract embeddings for each sentence. The embeddings are stored in `CEATDataCollection/embeddings` folder for all models.
### CEAT : Metric Calculation
The metric calculation is done using the following command:
```
python ./CeatDataCollection/CEATCalculations.py
```
This will take data from `CEATDataCollection/results/results_trait.pkl` file and calculate the metrics for each model. The metrics are stored in `CEATDataCollection/results` folder for all models.
### Logprob : Data Extraction
Extracting the values for log probability bias score can be done using the following command:
```
python ./bias_pred_score.py
```
This will take data from `data\positive_traits_bn.csv` and `data\negative_traits_bn.csv` files and calculate the log probability bias score for each model. The scores are stored in `results` folder for definite model.
### Logprob : Metric Calculation
Further analysis of data are found in `Notebooks/Exposing_Bias_in_BanglaLanguageModels.ipynb` and `Notebooks/Data_Analytics_for_bangla_bias_detection.ipynb` notebooks.

