from os.path import join, dirname, basename
import copy
import glob
import argparse
import pandas as pd

# setup parser
parser = argparse.ArgumentParser()
parser.add_argument("result_root",
                    help="root directory where all the result file (.csv) dumped")
parser.add_argument("out_fn", default='result.csv',
                    help="output result file (.csv) name")
args = parser.parse_args()

design = pd.read_csv('./eval/data/x.csv')
design.columns = ['self', 'bpm', 'year', 'taste', 'tag', 'lyrics', 'cdr_tag', 'artist', 'arc', 'n']

ARC = {
    1: 'MSSCR',
    2: 'MSCR@2',
    3: 'MSCR@4',
    4: 'MSCR@6',
    5: 'MSSR@fc'
}
design.arc = design.arc.map(ARC)

# add single sources
r = [0, 0, 0, 0, 0, 0, 0, 0, 'SSR', 1]  # row template
for i, name in enumerate(design.columns[:8]):
    r_ = copy.deepcopy(r)
    r_[i] = 1

    row = pd.DataFrame([r_], columns=design.columns)
    row.index = [name]

    design = design.append(row)

# adding all sources
r = [1, 1, 1, 1, 1, 1, 1, 1, None, 8]  # row template
for i, case in ARC.items():
    r_ = copy.deepcopy(r)
    r_[-2] = case

    if case == 'MSSCR':  # pure-concatenation
        row = pd.DataFrame([r_], columns=design.columns)
        row.index = ['allsrcnull']

    elif case == 'MSCR@2':  # branch at 2
        row = pd.DataFrame([r_], columns=design.columns)
        row.index = ['allsrc2']

    elif case == 'MSCR@4':  # branch at 4
        row = pd.DataFrame([r_], columns=design.columns)
        row.index = ['allsrc4']

    elif case == 'MSCR@6':  # branch at 6
        row = pd.DataFrame([r_], columns=design.columns)
        row.index = ['allsrc6']

    elif case == 'MSSR@fc':  # branch at fc
        row = pd.DataFrame([r_], columns=design.columns)
        row.index = ['allsrcfc']

    design = design.append(row)

# adding baselines
r_ = [0, 0, 0, 0, 0, 0, 0, 0, 'null', 'null']
for baseline_name in ['choi', 'mfcc', 'random']:
    row = pd.DataFrame([r_], columns=design.columns)
    row.index = [baseline_name]
    design = design.append(row)


design = design.reset_index()
design.columns = ['id'] + list(design.columns[1:])


# load the data
fns = glob.glob(join(args.result_root, '*/*.csv'))

# metric map
METRIC = {
    'classification': 'accuracy',
    'regression': 'r^2'
}

# task map
TASK = {
    'FMA': 'classification',
    'GTZAN': 'classification',
    'IRMAS': 'classification',
    'eBallroom': 'classification',
    'MusicEmoArousal': 'regression',
    'MusicEmoValence': 'regression',
    'Jam': 'recommendation',
    'Lastfm1k': 'recommendation'
}

SOURCE = {'bpm', 'tag', 'cdr', 'artist', 'self', 'year', 'lyrics', 'taste'}


def load_csv(fn):
    """"""
    d = pd.read_csv(fn)

    # 1. aggregate results from folds
    # 2. organize fields
    if d.iloc[0]['task'] == 'recommendation':
        # identify the dataset
        data = basename(dirname(fn))

        # get the mean & variance of each metric
        d3 = []
        for split in ['sparse', 'dense']:
            d0 = d[d['split'] == split]

            d1 = d0.groupby('metric').mean()

            if d.id.dtype == int:
                d1.columns = ['id', 'mean']
            else:
                d1.columns = ['mean']
                d1['id'] = d.iloc[0]['id']

            d1['std'] = d0.groupby('metric').std()['value']

            d2 = d1.reset_index()

            d2['data'] = '{}_{}'.format(data, split)
            d2['task'] = TASK[data]
            d2['model'] = d.iloc[0]['model']

            d3.append(d2)
        res = pd.concat(d3, axis=0)

    else:
        data = d.iloc[0]['task']

        # get the mean & variance of the metric (per model)
        d1 = d.groupby('model').mean()

        if d.id.dtype == int:
            d1.columns = ['id', 'mean']
        else:
            d1.columns = ['mean']
            d1['id'] = d.iloc[0]['id']

        d1['std'] = d.groupby('model').std()['value']

        res = d1.reset_index()

        res['data'] = data
        res['task'] = TASK[data]
        res['metric'] = METRIC[TASK[data]]

    # process id
    if res['id'].dtype == int:
        # res['id'] += 1
        # for the consistency with original design mat
        pass
    else:
        if res.iloc[0]['id'] in SOURCE:
            # fix cdr case
            if res.iloc[0]['id'] == 'cdr':
                res['id'] = 'cdr_tag'
            else:
                pass

        elif res.iloc[0]['id'] in {'random', 'mfcc', 'choi'}:
            res['id'] = res.iloc[0]['id']

        else:
            # cutoff allsrc trial number
            res['id'] = res.iloc[0]['id'][:-1]

        if res.iloc[0]['id'] == 'self_':
            res['id'] = 'self'

    return res

D = pd.concat(map(load_csv, fns), axis=0)

# join the design matrix
result = D.set_index('id').join(design.set_index('id'), how='inner')

# organize columns
result = result[
    ['data', 'task', 'model', 'metric', 'mean', 'std',
     'self', 'bpm', 'year', 'taste', 'tag' , 'lyrics' ,'cdr_tag', 'artist',
     'arc', 'n']
]

# save
result.to_csv(args.out_fn)
