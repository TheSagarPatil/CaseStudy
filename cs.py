"""
@Author : Sagar Patil, github.com/thesagarpatil
@Date : 2021/03/07

@Topic : implement random forest algorithm to predict Breast cancer
"""

#import packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pdb

#file path
INPUT_PATH = "dataset.csv"
OUTPUT_PATH = "data.csv"
HEADERS = ["CodeNumber",
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses",
    "CancerType"
]

#read data

def read_data(path):
    """
    read_data
    return: dataset
    """
    data = pd.read-csv(path)
    return data

#get data

def get_headers(dataset):
    """
    get headers 
    return : datasetColumns
    """
    return dataset.columns.values

def add_headers(dataset, headers):
    """
    add headers to dataset
    return : modifiedDataset
    """

def data_file_to_csv():
    """
    return : none
    """
    headers = HEADERS
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, headers)
    dataset.to_csv(OUTPUT_PATH, index=False)

def split_dataset(dataset, train_percentage, feature_headers, target_headers):
    """
    param : dataset, 
        train_percentage, 
        feature_headers, 
        target_headers
    return : train_x, test_x, train_y, test_y
    """
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_headers], train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def dataset_statistics(dataset):
    """
    return: none; describe dataset
    """
    print(dataset.describe())
def handle_missing_values(dataset, missing_values_header, missing_label):
    """
    filter missing values from dataset i.e. wrt label
    """
    return dataset[dataset[missing_values_header] != missing_label]
    
def random_forest_classifier(features, target):
    """
    return :fit dataset with features, data
    """
    clf = RandomForestClassifier()
    return clf.fit(features, target)

def main():
    #load data
    dataset = pd.read_csv(INPUT_PATH)
    #show statistics
    dataset_statistics(dataset)
    dataset = handle_missing_values(dataset,HEADERS[6], '?')
    #split dataset
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    
    #getDetails of dataset
    print("train_X Shape", train_x.shape)
    print("test_x Shape", test_x.shape)
    print("train_y Shape", train_y.shape)
    print("test_y Shape", test_y.shape)

    
    trained_model = random_forest_classifier(train_x, train_y)
    print("trained_model", trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0,5,1):
        print("actual outcome :: {} and predicted Outcome {}".format(list(test_y)[i], predictions[i]))

    print("train_accuracy", accuracy_score(train_y, trained_model.predict(train_x)) * 100)
    print("train_accuracy", accuracy_score(test_y, predictions) * 100)
    print("confussion matrix", confusion_matrix(test_y, predictions))
if (__name__ == "__main__"):
    print("Breast cancer case study")
    main()