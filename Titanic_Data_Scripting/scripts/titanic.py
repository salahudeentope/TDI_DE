import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import click
import pandas as pd
import numpy as np


class TitanicCleaner:
    """
    Data pipeline Class to consolidate all the methods 
    for loading, transforming, and cleaning the Titanic dataset 
    """

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)  # Load data once on initialization
        print(f"Loaded data from {csv_file}")
        print(f"These are the columns: {self.data.columns}")
        print('---------------------------------')
        print(f"This is the head: {self.data.head(2)}")

    def _data_info(self):
        """
        Function that provides initial info about the data
        """
        print(self.data.describe(include='all'))
        print(f"THIS IS DATA INFO: {self.data.info()}")

    def _fill_missing(self):
        """
        Function that fills missing values
        """
        print(self.data.isnull().sum())
        self.data.Age.fillna(0, inplace=True)
        self.data.Fare.fillna(1, inplace=True)
        self.data.Cabin.fillna("NA", inplace=True)
        print("Missing values filled.")
        return self.data

    def _drop_duplicates(self):
        """
        Function that removes duplicate rows
        """
        print(self.data.duplicated().value_counts())
        self.data = self.data.drop_duplicates()
        print("Duplicate rows dropped.")
        return self.data

    def _bin_age(self):
        """
        Bins the 'Age' column into categories using apply()
        """
        def age_group(age):
            if age < 18:
                return '<18'
            elif 18 <= age < 40:
                return '18-40'
            elif 40 <= age < 60:
                return '40-60'
            else:
                return '60+'
        
        self.data['AgeGroup'] = self.data['Age'].apply(lambda x: age_group(x))
        print(f"Binned Age Data:\n{self.data[['Age', 'AgeGroup']].head()}")
        return self.data

    def _family_size(self):
        """
        Adds 'FamilySize' column using apply()
        """
        self.data['FamilySize'] = self.data.apply(lambda row: row['SibSp'] + row['Parch'], axis=1)
        print(f"Family Size Added:\n{self.data[['SibSp', 'Parch', 'FamilySize']].head()}")
        return self.data

    def _map_embarked(self):
        """
        Maps the 'Embarked' column to readable names using apply()
        """
        embark_mapping = {'S': "Southampton", "C": "Cherbourg", 'Q': 'Queenstown'}
        self.data['Embarked_mapped'] = self.data['Embarked'].apply(lambda x: embark_mapping.get(x, x))
        print(f"Mapped Embarked Locations:\n{self.data[['Embarked', 'Embarked_mapped']].head()}")
        return self.data

    def clean_titanic(self, show_info=False, fill_missing=False, drop_duplicates=False, bin_age=False, family_size=False, map_embarked=False):
        """
        Function that cleans the dataset based on the flags
        """
        if show_info:
            self._data_info()
        if fill_missing:
            self.data = self._fill_missing()
        if drop_duplicates:
            self.data = self._drop_duplicates()
        if bin_age:
            self.data = self._bin_age()
        if family_size:
            self.data = self._family_size()
        if map_embarked:
            self.data = self._map_embarked()
        return self.data


# Click-based CLI
@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--show-info', is_flag=True, help="Prints the dataset info.")
@click.option('--fill-missing', is_flag=True, help="Fill missing values in the dataset.")
@click.option('--drop-duplicates', is_flag=True, help="Drop duplicate rows in the dataset.")
@click.option('--bin-age', is_flag=True, help="Bin age into groups.")
@click.option('--family-size', is_flag=True, help="Create family size column.")
@click.option('--map-embarked', is_flag=True, help="Map embarked column to readable names.")
@click.option('--show-head', is_flag=True, help="Show the first few rows of the cleaned dataset.")
def clean_csv(csv_file, show_info, fill_missing, drop_duplicates, bin_age, family_size, map_embarked, show_head):
    """CLI function to clean the Titanic dataset with optional transformations."""
    cleaner = TitanicCleaner(csv_file)

    # Clean the data based on selected options
    cleaner.clean_titanic(
        show_info=show_info,
        fill_missing=fill_missing,
        drop_duplicates=drop_duplicates,
        bin_age=bin_age,
        family_size=family_size,
        map_embarked=map_embarked
    )
    
    print("Data cleaning/transformation completed!")

    # Show the first few rows of the dataset if --show-head flag is passed
    if show_head:
        print("\nHere is the current state of the dataset after applying transformations:")
        print(cleaner.data.head())
        print(cleaner.data.columns)

if __name__ == '__main__':
    clean_csv()
