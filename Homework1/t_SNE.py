from sklearn.manifold import TSNE
import pandas as pd

data = pd.read_csv("E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit_handle.csv")
data.drop('ID', axis=1, inplace=True)
data.drop('Label', axis=1, inplace=True)
tsne_data = TSNE(learning_rate=100).fit_transform(data)

print(tsne_data.shape)
output = "E://Study//研1下//4大数据分析与应用//Homework//Work 1//t_SNE.csv"
pd.DataFrame(tsne_data).to_csv(output, sep=',', index=False)
