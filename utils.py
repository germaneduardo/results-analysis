import itertools
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines, markers

def get_markers(n):
    """ generate combinations of markers and linestyles"""
    l = [ll for ll in lines.lineStyles.keys() if ll is not None and (ll not in ('', 'None', ' '))]
    m = [mm for mm in markers.MarkerStyle.markers.keys() if mm is not None and (mm not in ('', 'None', ' '))]
    lms = list(itertools.product(m, l))
    samples = np.random.choice(len(lms),n)
    mm = []
    ll = []
    for idx in samples:
        mm.append(lms[idx][0])
        ll.append(lms[idx][1])
    return(mm, ll)

def read_results(folder, ignore_files, str_to_split_db="AllResults", db_idx = 0):
    """ read csv files into pandas dataframes"""
        
    path = pathlib.Path(folder) 
    results = pd.DataFrame()
    files = []
    for i in path.glob('**/*.csv'):
        if not i.name  in (ignore_files):
            partial = pd.read_csv(i.absolute())
            partial['database'] = i.name.split(str_to_split_db)[db_idx]
            results = pd.concat([results, partial], ignore_index = True)
    return(results)

def get_best_for(g, measures = None, measure_sort = 'F1', ascending=False, factorsModel= None, agg = False):
    """Aggregates to get the best results based on the a measure and some factorsmodel"""
    test = g.copy()
    test = test[test['type'] == 'Test']
    best = test.groupby(factorsModel)[measures].mean() if agg  else test.groupby(factorsModel)[measures].mean()
    best.sort_values(by = measure_sort, ascending = ascending, inplace= True)
    best['type'] = 'Test'
    params = best.head(1).copy().reset_index()[factorsModel].to_dict('records')[0]
    # train
    params['type'] = 'Train'
    train = g.loc[(g[list(params)] == pd.Series(params)).all(axis=1)].copy()

    tr_best = train.groupby(factorsModel)[measures].mean()
    tr_best.sort_values(by = measure_sort, ascending = ascending, inplace= True) 
    tr_best['type'] = 'Train'
    return(pd.concat([best.head(1), tr_best.head(1)]))

def increase_lw(g, lw = 2.8):
    """ increase the linewidth of all the lines in seaborn plot"""
    for ax in np.ravel(g.axes):
        for l in ax.lines:
            plt.setp(l,linewidth=lw)