import pandas as pd
import math

def calc_distance(x):
    A = [0.276,	0.619] # replace this with your A values
    return math.sqrt(((A[0]-x['x'])**2)+((A[1]-x['y'])**2))

df = pd.read_csv('input.csv')
df['distance'] = df.apply(calc_distance, axis=1)
df = df.sort_values(by=['distance'])
df.to_csv('input_py.csv')
