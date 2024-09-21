"""
Data Pipeline and CLI app for the Titanic Dataset
"""
required_packages = ["numpy", 'seaborn', "pandas", 'click', 'argparse']

# Function to install required packages.

def install_packages(packages):
    for package in packages:
        import importlib
        try:
            importlib.import_module(package)
        except ImportError:
            import pip
            pip.main(['install', package])
        # finally:
        #     globals()[package] = importlib.import_module(package)

install_packages(required_packages)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click



class TitanicCleaner ():
    """
    Data pipeline Class to consolidate all the methods 
    for loading, transforming and cleaning the Titanic dataset 
    """

    def __init__ (self, data):
        self.data = data

    def _load_data (self):
        
        """
        Function that loads the titanic dataset
        """
        data = pd.read_csv(self.data)
        print(f"These are the columns: {data.columns}")
        print('---------------------------------')
        print(f"This is the head: {data.head(2)}")
        print(data.tail(2))
        return data

    def _data_info (self, data):
        """
        Function that provides intial info about the  data
        """
        print(data.describe(include='all'))
        print(f"THIS IS DATA INFO: {data.info()}")

    def _fill_missing(self, data):
        """
        Function that fills missing values
        """
        print(data.isnull().sum())
        data.Age.fillna(0, inplace=True)
        data.Fare.fillna(1, inplace=True)
        data.Cabin.fillna("NA", inplace=True)
        return data

    def _drop_duplicates(self, data):
        """
        Function that removes duplicate rows
        """
        print(data.duplicated().value_counts())  
        data = data.drop_duplicates()
        return data

    def _bin_age (self, data):
        bins = [0, 18, 40, 60, np.inf]
        names = ['<18', '18-40', '40-60', '60+']
        data['AgeGroup'] = pd.cut(data['Age'], bins, labels=names)
        return data

    def _family_size(self, data):
        data['FamilySize'] = data['SibSp'] + data['Parch']
        return data

    def _map_embarked (self, data):
        embark = {'S': "Southampton", "C": "Cherbourg", 'Q': 'Queenstown'}
        data = data.replace({"Embarked": embark})
        print(data.groupby(['Embarked'])['Survived'].sum())
        return data

    def clean_titanic(self):
        """
        Function that cleans the dataset by combining all the defined data cleaning functions
        """
        data = self._load_data()
        self._data_info(data)
        if data.isnull().sum().sum() > 1:
            data = self._fill_missing(data)
        data = self._drop_duplicates(data)
        data = self._bin_age(data)
        data = self._family_size(data)
        data = self._map_embarked(data)
        return data

    

if __name__=='__main__':
    cleaner = TitanicCleaner("data/titanic.csv")
    data = cleaner.clean_titanic()

    print(data.columns)
