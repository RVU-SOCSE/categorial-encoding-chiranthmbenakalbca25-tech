import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df = pd.read_csv("C:/Users/manub/Desktop/CHIRU/PYTHON/laptop.csv")
print(df.head())


categorical_cols = df.select_dtypes(include=['object', 'string']).columns
print("\nCategorical Columns:\n", categorical_cols)


label_encoded_df = df.copy()
le = LabelEncoder()

for col in categorical_cols:
    label_encoded_df[col] = le.fit_transform(label_encoded_df[col])

print("\nDataset after Label Encoding:\n")
print(label_encoded_df.head())

ohe = OneHotEncoder(sparse_output=False, drop='first')

encoded_data = ohe.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=ohe.get_feature_names_out(categorical_cols)
)

numerical_df = df.drop(columns=categorical_cols)
final_df = pd.concat([numerical_df, encoded_df], axis=1)

print("\nDataset after One-Hot Encoding:\n")
print(final_df.head())
