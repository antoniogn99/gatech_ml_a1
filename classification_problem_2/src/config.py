import os
from os.path import join


DIRECTORY = join(os.getcwd(), "classification_problem_2")
DATA_PATH = join(DIRECTORY, "data")
ORIGINAL_DATA_CSV_PATH = os.path.join(DATA_PATH, "Loan_Default.csv")
DATA_CSV_PATH = os.path.join(DATA_PATH, "c2_data.csv")