import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE

df = pd.read_csv("dataset\\train.csv")
print(df.shape)

tsne = TSNE(n_components=2,verbose=1,random_state=123)
z = tsne.fit_transform(df.iloc[:,1:])

df['x'] = z.T[0]
df['y'] = z.T[1]

fig = px.scatter(df,x="x",y="y",color="label")
fig.show()

