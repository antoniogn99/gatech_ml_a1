import pandas as pd
import numpy as np

from config import ORIGINAL_DATA_CSV_PATH


def prepare_data():
    df = pd.read_csv(ORIGINAL_DATA_CSV_PATH)
    df.columns= df.columns.str.lower()
    df.drop(columns=["customerid"], inplace=True)

    # Replace space by -1 and convert to float
    df["totalcharges"] = df["totalcharges"].replace(' ', -1)
    df["totalcharges"] = df["totalcharges"].astype(float)
    
    # Encode binary variables
    df = df.replace(["Male", "Female", "Yes", "No"], [1, 0, 1, 0])

    # Create list with the categorical variables
    cat_vars = []
    for var, var_type in zip(df.dtypes.index, df.dtypes):
        if var_type == 'object':
            cat_vars.append(var)

    # Encode categorical variables using One Hot Encoder
    for var in cat_vars:
        dummies = pd.get_dummies(df[var], prefix=var)
        df.drop(var, axis=1, inplace=True)
        df = pd.merge(
            left=df,
            right=dummies,
            left_index=True,
            right_index=True,
        )
    
    df.to_csv("c2_data.csv", index=False)


if __name__ == "__main__":
    prepare_data()