import itertools
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines, markers
import seaborn as sns

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

def get_best_for(g, measures = None, measure_sort = 'F1', ascending=False, factorsModel= None, agg = False, debug = False) :
    """Aggregates to get the best results based on the a measure and some factorsmodel"""
    test = g.copy()

    test = test[test['type'] == 'test']
    best = test.groupby(factorsModel)[measures].mean()
    best.sort_values(by = measure_sort, ascending = ascending, inplace= True)
    if (debug):
        print(best.head(1))
    params = best.head(1).copy().reset_index()[factorsModel].to_dict('records')[0]
    params_train = params
    params_test = params
    params_train['type'] = 'train'
    params_test['type'] = 'test'    
    train = g.loc[(g[list(params)] == pd.Series(params)).all(axis=1)].copy()
    test = g.loc[(g[list(params)] == pd.Series(params)).all(axis=1)].copy()
    if agg:
        tr_best = train.groupby(factorsModel)[measures].mean().sort_values(by = measure_sort, ascending = ascending, inplace= False) 
        tr_best['type'] = 'train'
        return(pd.concat([best.head(1), tr_best.head(1)]))
    else:
        train = train.loc[:, factorsModel+ measures]
        train['type'] = 'train'
        test = test.loc[:, factorsModel+ measures]
        test['type'] = 'test'
        return(pd.concat([train, test], ignore_index=True, axis = 0))

def increase_lw(g, lw = 2.8):
    """ increase the linewidth of all the lines in seaborn plot"""
    for ax in np.ravel(g.axes):
        for l in ax.lines:
            plt.setp(l,linewidth=lw)
            

            
"""
OTHER OLDER useful methods
"""

aspect = 1.5
sns.set(font_scale=1.3, style='white')

def get_db_from_ax(ax) :
    db_str = ax.title.get_text()
    db = db_str.split(" | ")[0].split("=")[1].strip()
    return (db)
 
def draw_reference (g, df_reference, measure,  ls = {'Train': '--', 'Test': '-'}, lw = 0.1):
    axs  =np.ravel (g.axes)
    for ax in np.ravel(axs):
        db = get_db_from_ax(ax)
        mea_train = df_reference.loc[(df_reference.database==db) & (df_reference.type == 'Train'), measure]
        mea_test = df_reference.loc[(df_reference.database==db) & (df_reference.type == 'Test'), measure]
        if len(mea_train) > 0:
            ax.axhline(mea_train.values[0], ls = ls['Train'], color = 'k', lw = lw,   label = 'Train', alpha = 0.9)
            ax.axhline(mea_test.values[0], ls = ls['Test'], color = 'k', lw = lw,   label = 'Test', alpha = 0.9)
        else:
            print("referece for db " + db + " missing")

def get_result_plot(df, measure, cond, df_reference = None):
    
    if type(measure) == tuple:
        measure_show = measure[0]
        measure_for_best =  measure[1]
    else:
        measure_show,measure_for_best = measure, measure

    data = (df[cond].groupby(factorsIS)
                .apply(lambda g : ut.get_best_for(g, measures, measure_sort = measure_for_best, factorsModel = factorsModel))
                .reset_index())
    data['is'] = data['one_class_method']

    g= sns.relplot(x="ands", y=measure_show, hue = 'lshMEthod',
            style = 'type', 
            aspect=aspect, 
            row = 'database', 
            col = 'is',
            kind = 'line',
            data=data,
            legend = 'full', 
            markers = True, facet_kws = {'sharey': 'row'})
    if df_reference is not None:
        draw_reference(g, df_reference, measure)
    ut.increase_lw(g)
    plt.show()
    g.savefig(f"report/{measure}.png")
    return(g)
