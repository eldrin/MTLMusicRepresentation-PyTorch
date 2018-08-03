class Config:
    """ global setup for all experiments """
    SR = 22050  # sampling rate
    N_FFT = 1024  # number of fft window size
    HOP_LEN = 256  # hop size for STFT
    MONO = False  # number of channels used
    N_CH = 1 if MONO else 2  # integer equivalent for above one
    N_STEPS = 216  # number time steps (corresponding to 2.5 sec with current setup)
    N_BINS = 128  # number of mel bins
    L2 = 1e-7  # L2 regularization coefficient
    LEARN_RATE = 0.0005  # learning rate for ADAM optimizer
    N_EPOCHES = 1000  # number of epoches
    BATCH_SZ = 256  # batch size