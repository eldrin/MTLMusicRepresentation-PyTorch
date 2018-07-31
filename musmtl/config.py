class Config:
    """ global setup for all experiments """
    SR = 22050  # sampling rate
    N_FFT = 1024  # number of fft window size
    HOP_LEN = 256  # hop size for STFT
    MONO = False  # number of channels used
    N_CH = 1 if MONO else 2  # integer equivalent for above one
    N_STEPS = 216  # number time steps (corresponding to 2.5 sec with current setup)
    N_BINS = 128  # number of mel bins