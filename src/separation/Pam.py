import librosa
import numpy as np

def init_W0(dir_path,n_fft=1024):    
    print("Loading dictionnary...")
    dico_piano , sr = librosa.load(dir_path+'/dico piano Piano élec.wav')
    dico_batterie_up , _ = librosa.load(dir_path+'/dico batterie Batterie up.wav')
    dico_batterie_kick , _ = librosa.load(dir_path+'/dico batterie Batterie kick.wav')
    dico_saxo , _ = librosa.load(dir_path+'/dico saxo Saxophone.wav')
    dico_basse , _ = librosa.load(dir_path+'/dico basse Basse.wav')
    dico_guitare , _ = librosa.load(dir_path+'/dico guitare Guitare élec.wav')

    dico_batterie = dico_batterie_up + dico_batterie_kick

    dico_guitare = dico_guitare[2*sr:]
    dico_piano = dico_piano[2*sr:]
    dico_batterie = dico_batterie[2*sr:]
    dico_saxo = dico_saxo[2*sr:]
    dico_basse = dico_basse[2*sr:]

    wav_dict = {'piano': dico_piano, 'batterie': dico_batterie, 'saxo': dico_saxo, 'basse': dico_basse, 'guitare': dico_guitare}

    
    def remove_low_energy(key,samples,threshold):
        waveform = wav_dict[key]
        returned_samples = []
        for i in range(len(samples)-1):
            note = waveform[samples[i]:samples[i+1]-1000]
            if np.sum(note**2) > threshold:
                returned_samples.append(samples[i])
        returned_samples.append(samples[-1])
        return returned_samples
    
    def onset_graph(key:str,o_threshold:float = 5,e_threshold = 2):
        waveform = wav_dict[key]
        D = np.abs(librosa.stft(waveform))
        o_env = librosa.onset.onset_strength(y=waveform, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        for i,frame in enumerate(onset_frames):
            if o_env[frame-2:frame+2].max()<o_threshold:
                onset_frames[i] = -1
        onset_frames_new = onset_frames[onset_frames!=-1]
        samples = librosa.frames_to_samples(onset_frames_new)
        samples = remove_low_energy(key,samples,e_threshold)
        onset_frames_new = librosa.samples_to_frames(samples)
        if key == 'guitare':
            onset_frames_new = onset_frames_new[:-1]
        return samples
    
    piano_onsets = onset_graph('piano',e_threshold=0.5)
    drums_onsets = onset_graph('batterie')
    saxo_onsets = onset_graph('saxo')
    basse_onsets = onset_graph('basse',9)
    guitare_onsets = onset_graph('guitare',4.6)

    W = np.random.rand(5, n_fft//2+1, 52)
    
    def note_generator(waveform,onsets):
        onsets = np.concatenate((onsets,[len(waveform)]))
        for i in range(len(onsets)-1):
            note = waveform[onsets[i]:onsets[i+1]-1000]
            if len(note)>5*sr: 
                note = note[:5*sr]
            yield note

    for i_instr , generator in enumerate(map(note_generator,
                                            [wav_dict['piano'],wav_dict['batterie'],wav_dict['saxo'],wav_dict['basse'],wav_dict['guitare']],
                                            [piano_onsets,drums_onsets,saxo_onsets,basse_onsets,guitare_onsets])):
        for i,note in enumerate(generator):
            fft = np.fft.fft(note,n_fft)[:n_fft//2+1] / (n_fft*2)         
            fft/=np.max(np.abs(fft))    
            W[i_instr,:,i] = np.abs(fft)

    return W

if __name__=="__main__":
    W = init_W0("../piste_pam")
    print(W.shape)