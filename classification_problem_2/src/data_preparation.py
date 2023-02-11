import pandas as pd

from config import ORIGINAL_DATA_CSV_PATH

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    df = pd.read_csv(ORIGINAL_DATA_CSV_PATH)
    df.columns= df.columns.str.lower()
    df.drop(columns=['id', "year"], inplace=True)

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
    
    # Replace NaN by -1
    df.fillna(df.mean(), inplace=True)
    df = df[:10000]
    df.to_csv("c2_data.csv", index=False)


if __name__ == "__main__":
    prepare_data()