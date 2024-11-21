import bnlearn as bn
import polars as pl
from distfit import distfit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout

exampleData = pl.DataFrame(bn.import_example("auto_mpg"))

# Define horsepower bins based on domain knowledge
bins = [0, 100, 150, exampleData['horsepower'].max()]
labels = ['min','low', 'medium', 'high','max']

# discretise with a probability distribution
colsToAdd = []

for col in ['acceleration','mpg', 'displacement', 'weight']:
    print(col)
# set a CI (the break will be the chosen quantiles 
# so shouldnt be all inclusive or all will end up in the same bin)
    dist = distfit(alpha=0.1)

# polars Series are not based on numpy arrays so need to transform
    dist.fit_transform(exampleData.select(col).to_numpy())

# dist.model doesent store median easily so would need to find it from upper-lower/2 
    contBins = [exampleData[col].min(), 
                dist.model['CII_min_alpha'], 
                dist.model['CII_max_alpha'], 
                exampleData[col].max()]
    print(contBins)    
    
    if(col == 'acceleration'):
        contLabels = ['superfast','fast', 'normal', 'slow','dead']
    else:
        contLabels = ['min','low', 'medium', 'high','max']
    
    print(contLabels)    
    
    colsToAdd.append(exampleData[col]
                             .cut(breaks=contBins, labels=contLabels)
                             .alias('cat' + col) )  
    
    # # delete old columns
    exampleData = exampleData.select(pl.all().exclude(col))

 
exampleData = exampleData.with_columns(colsToAdd)

# assigning Series to a new column (different in polars).
# with_columns adds the column, alias renames the column
exampleData = (exampleData
               .with_columns(exampleData['horsepower']
                             .cut(breaks=bins, labels=labels)
                             .alias('catHorsepower')
                             )
)

# delete old columns
exampleData = exampleData.select(pl.all().exclude('horsepower'))
dfNames = exampleData.columns
exampleData.head()
    
exampleData = pd.DataFrame(exampleData)
exampleData.columns = dfNames

model = bn.structure_learning.fit(exampleData, methodtype='hc')

# Compute edge strength
model = bn.independence_test(model, exampleData)

# Make plot and put the -log10(pvalues) on the edges
bn.plot(model, edge_labels='pvalue')

bn.plot_graphviz(model, edge_labels='pvalue')
bn.plot(model, interactive=True)
