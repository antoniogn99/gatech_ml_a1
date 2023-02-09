import os
from os.path import join


DIRECTORY = join(os.getcwd(), "classification_problem_1")
DATA_PATH = join(DIRECTORY, "data")
TRAIN_CSV_PATH = os.path.join(DATA_PATH, "c1_train.csv")
TEST_CSV_PATH = os.path.join(DATA_PATH, "c1_test.csv")