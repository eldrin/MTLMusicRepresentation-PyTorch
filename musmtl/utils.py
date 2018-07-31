from os.path import join, dirname, basename
import shutil
import pickle as pkl
import torch


def load_pickle(fn):
    """ python2 -> python3 loadable pickle loader
    
    Args:
        fn (str): path to pickle file

    ref: https://blog.modest-destiny.com/posts/python-2-and-3-compatible-pickle-save-and-load/
    """ 
    try:
        with open(fn, 'rb') as f:
            data = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(fn, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', fn, ':', e)
        raise
    return data


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """"""
    torch.save(state, filename)
    if is_best:
        new_basename = basename(filename)
        new_basename = new_basename.replace('checkpoint', 'model_best')
        new_fn = join(dirname(filename), new_basename)
        shutil.copyfile(filename, new_fn)

        
def log_amplitude(x):
    """"""
    log_spec = 10 * np.log(np.maximum(x, 1e-10))/np.log(10)
    log_spec = log_spec - np.max(log_spec)  # [-?, 0]
    log_spec = np.maximum(log_spec, -96.0)  # [-96, 0]

    return log_spec