import pandas as pd
import umap

data = pd.read_csv("E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit_handle.csv")
data.drop('ID', axis=1, inplace=True)
data.drop('Label', axis=1, inplace=True)
umap_data = umap.UMAP()
embedding = umap_data.fit_transform(data)
print(embedding.shape)
output = "E://Study//研1下//4大数据分析与应用//Homework//Work 1//umap.csv"
pd.DataFrame(embedding).to_csv(output, sep=',', index=False)
