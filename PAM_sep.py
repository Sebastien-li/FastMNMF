# %%
import os
import librosa
import numpy as np

from src.separation.FastMNMF2 import FastMNMF2
from Base import MultiSTFT


# %%
dir_path = "../wav_piste_pam"

# %%
mic_list = ['Basse','Batterie kick','Guitare élec','Piano élec','Batterie up','Saxophone']
list_config = []
for file in os.listdir(dir_path):
    add = True
    for mic in mic_list+['salle A',"Salle B","dico","mixture"]:
        if mic in file:
            add = False
    if add: list_config.append(file)

# %%
dict_mixtures = {}
for config in list_config:
    mixtures = []
    for mic in mic_list:
        config_split = config.split('.')
        config_split[0] += ' '+mic
        filename = '.'.join(config_split)
        waveform, sr = librosa.load(dir_path+"/"+filename)
        mixtures.append(waveform)
    multichannel_sound = np.array(mixtures)
    dict_mixtures[config] = multichannel_sound

# %%
dict_path = "../wav_piste_pam" 
gpu = 1
n_fft = 1024
n_source = 5
n_basis = 52
n_iter_init = 30
init_SCM = "circular"
g_eps = 5e-2
n_mic = 6
n_bit = 64
algo = "IP"
n_iter = 1000

# %%
if gpu < 0:
    import numpy as xp
else:
    try:
        import cupy as xp
        print("Use GPU " + str(gpu))
        xp.cuda.Device(gpu).use()
    except ImportError:
        print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU.\n")
        import numpy as xp

def run(i,dict_use):
    if dict_use:
        pdict_path = dict_path
        str_dict = " with "
    else:
        pdict_path = ""
        str_dict = " no "

    input_fname = list_config[i]
    wav = dict_mixtures[input_fname].T
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=n_fft)

    separater = FastMNMF2(
        n_source=n_source,
        dict_path=pdict_path,
        n_basis=n_basis,
        xp=xp,
        init_SCM=init_SCM,
        n_bit=n_bit,
        algo=algo,
        n_iter_init=n_iter_init,
        g_eps=g_eps,
    )

    separater.file_id = (input_fname.split("/")[-1].split(".")[0]+str_dict+"dict")
    separater.load_spectrogram(spec_FTM, sr)

    separater.solve(
        n_iter=n_iter,
        save_dir="results_separation",
        save_likelihood=True,
        save_param=False,
        save_wav=True,
        interval_save=5,
    )


for i in range(len(list_config)):
    run(i,dict_use=True)
    run(i,dict_use=False)

