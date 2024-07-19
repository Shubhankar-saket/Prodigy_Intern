
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv("train.csv")
df
reg=linear_model.LinearRegression()
reg.fit(df[['LotArea','BedroomAbvGr','FullBath','HalfBath']],df.SalePrice)
new_lot_prices=np.array([[1000,3,1,1]])
rp=reg.predict(new_lot_prices)
rp
