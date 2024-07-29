# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest calssifier with the default hyperparameters
## Intended Use
Binary classification model for individuals' salary with 2 categories of salaries: >=50k & <50k
## Training Data
Data includes both individuals' categorical features such as education level and numerical features such as age and their salaries as the target variable

## Evaluation Data
Data is divided using the scikit learn train test split and the test data is used for the evaluation

## Metrics
F1 score, precision and recall were used as the evaluation metrics. 
F1 score: 0.6049083995852056
Precision: 0.6679389312977099
Recall: 0.5527479469361971

## Ethical Considerations
The dataset has some sensitive information such as the gender that needs to be processed according to the regulations

## Caveats and Recommendations
This is a very basic version of the model. Better feature engineering and further fine tuning should be done for obtaining better results.
