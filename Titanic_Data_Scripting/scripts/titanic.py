import warnings
import os
import logging


# Suppress all warnings
warnings.filterwarnings("ignore")

import click
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("titanic_cleaner.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()


class TitanicCleaner:
    """
    Data pipeline Class to consolidate all the methods 
    for loading, transforming, and cleaning the Titanic dataset 
    """

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)  # Load data once on initialization
        logger.info(f"Loaded data from {csv_file}")
        logger.info(f"These are the columns: {self.data.columns.tolist()}")
        logger.info(f"This is the head: \n{self.data.head(2)}")

    def _data_info(self):
        """Provide initial info about the data."""
        logger.info("Data description:")
        logger.info(f"{self.data.describe(include='all')}")
        logger.info(f"Data info: {self.data.info()}")

    def _fill_missing(self):
        """Fill missing values."""
        logger.info("Checking for missing values:")
        logger.info(f"{self.data.isnull().sum()}")
        self.data.Age.fillna(0, inplace=True)
        self.data.Fare.fillna(1, inplace=True)
        self.data.Cabin.fillna("Unknown", inplace=True)
        logger.info("Missing values filled.")
        return self.data

    def _drop_duplicates(self):
        """Remove duplicate rows."""
        logger.info("Checking for duplicates:")
        logger.info(f"{self.data.duplicated().value_counts()}")
        self.data = self.data.drop_duplicates()
        logger.info("Duplicate rows dropped.")
        return self.data

    def _bin_age(self):
        """Bin the 'Age' column into categories."""
        def age_group(age):
            if age < 18:
                return '<18'
            elif 18 <= age < 40:
                return '18-40'
            elif 40 <= age < 60:
                return '40-60'
            else:
                return '60+'
        
        self.data['AgeGroup'] = self.data['Age'].apply(age_group)
        logger.info(f"Binned Age Data:\n{self.data[['Age', 'AgeGroup']].head()}")
        return self.data

    def _family_size(self):
        """Add 'FamilySize' column."""
        self.data['FamilySize'] = self.data.apply(lambda row: row['SibSp'] + row['Parch'], axis=1)
        logger.info(f"Family Size Added:\n{self.data[['SibSp', 'Parch', 'FamilySize']].head()}")
        return self.data

    def _map_embarked(self):
        """Map the 'Embarked' column to readable names."""
        embark_mapping = {'S': "Southampton", "C": "Cherbourg", 'Q': 'Queenstown'}
        self.data['Embarked_mapped'] = self.data['Embarked'].apply(lambda x: embark_mapping.get(x, x))
        logger.info(f"Mapped Embarked Locations:\n{self.data[['Embarked', 'Embarked_mapped']].head()}")
        return self.data

    def save_data(self, temp_file='temp_titanic_data.csv'):
        """Save the current state of the data to a temporary CSV file."""
        self.data.to_csv(temp_file, index=False)
        logger.info(f"Current state of the data saved to {temp_file}.")

    def load_data(self, temp_file='temp_titanic_data.csv'):
        """Load the current state of the data from the temporary CSV file."""
        if os.path.exists(temp_file):
            self.data = pd.read_csv(temp_file)
            logger.info(f"Loaded current state of the data from {temp_file}.")
        else:
            logger.warning(f"No temporary data file found at {temp_file}.")


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

    # Load current state from the temporary file if it exists
    cleaner.load_data()

    # Clean the data based on selected options
    if show_info:
        cleaner._data_info()
    if fill_missing:
        cleaner._fill_missing()
    if drop_duplicates:
        cleaner._drop_duplicates()
    if bin_age:
        cleaner._bin_age()
    if family_size:
        cleaner._family_size()
    if map_embarked:
        cleaner._map_embarked()

    # Save the current state of the data
    cleaner.save_data()

    logger.info("Data cleaning/transformation completed!")

    # Show the first few rows of the dataset if --show-head flag is passed
    if show_head:
        logger.info("Here is the current state of the dataset after applying transformations:")
        logger.info(f"{cleaner.data.head()}")
        logger.info(f"Columns: {cleaner.data.columns.tolist()}")

if __name__ == '__main__':
    clean_csv()
