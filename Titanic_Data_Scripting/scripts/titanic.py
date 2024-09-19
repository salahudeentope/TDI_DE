# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# import numpy as np
# import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.ticker import StrMethodFormatter

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

required_packages = ["numpy", 'seaborn', "pandas"]

install_packages(required_packages)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_titanic(file: str = "./data/titanic.csv"):
    """
    Function that cleans titanic
    """
    data = pd.read_csv(file)
    print(f"These are the columns: {data.columns}")
    print('---------------------------------')
    print(f"This is the head: {data.head(2)}")
    print(data.tail(2))
    print(data.isnull().sum())
    if data.isnull().sum().sum() > 1:
        data = fill_missing(data)
    print(f"THIS IS DATA INFO: {data.info()}")
    print(data.duplicated().value_counts())
    data.drop_duplicates
    print(data.describe(include='all'))
    print(data.groupby(['Pclass', 'Sex'])[['Embarked', 'Survived',]].value_counts())

def fill_missing(data):
    data.Age.fillna(0, inplace=True)
    data.Fare.fillna(1, inplace=True)
    data.Cabin.fillna("NA", inplace=True)
    return data

if __name__=='__main__':
    clean_titanic()
