import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib.pylab import *
import statsmodels.tsa.stattools as ts;
data=pd.DataFrame({"A":[1,2,3,4]})

def dun(x):
	print(x,"*")

op=data["A"].agg(dun)

