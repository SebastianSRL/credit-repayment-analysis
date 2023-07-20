# Home Credit Repayment Prediction: Binary Classification with AUC Evaluation
> Home Credit Default Risk

## The Business problem

We want to predict whether the person applying for a home credit will be able to repay their debt or not. Our model will have to predict a 1 indicating the client will have payment difficulties: he/she will have late payment of more than X days on at least one of the first Y installments of the loan in our sample, 0 in all other cases.

We will use [Area Under the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419) as the evaluation metric, so our models will have to return the probabilities that a loan is not paid for each input data.

## About the data

The original dataset is composed of multiple files with different information about loans taken. In this project, we will work exclusively with the primary files: `application_train_aai.csv` and `application_test_aai.csv`.

You don't have to worry about downloading the data, it will be automatically downloaded from the `sprint_project.ipynb` notebook in `Section 1 - Getting the data`.

## Technical aspects

To develop this Machine Learning model you will have to primary interact with the Jupyter notebook provided, called `sprint_project.ipynb`.

The technologies involved are:
- Python as the main programming language
- Pandas for consuming data from CSVs files
- Scikit-learn for building features and training ML models
- Matplotlib and Seaborn for the visualizations
- Jupyter notebooks to make the experimentation in an interactive way

## Installation

A `requirements.txt` file is provided with all the needed Python libraries for running this project. For installing the dependencies just run:

```console
$ pip install -r requirements.txt
```

*Note:* We encourage you to install those inside a virtual environment.

## Tests

We provide unit tests along with the project that you can run and check from your side the code meets the minimum requirements of correctness needed to approve. To run just execute:

```console
$ pytest tests/
```
