import score_informed_nmf as nmf
import mido
import sklearn.decomposition
import librosa
import numpy as np
import os
from typing import NamedTuple, Optional, List

datasets = ['chorales_synth']
source_names = ['soprano', 'alto', 'tenor', 'bass']

class NMFConfiguration(NamedTuple):
    config_name: str
    num_partials: Optional[int]
    phi: float
    tol_on: float
    tol_off: float
    n_fft: int
    mask: bool = True

def separate(chorale, dataset, config: NMFConfiguration):
    output_dir = f'nmf_evaluation/{dataset}/{config.config_name}/{chorale}'
    os.makedirs(output_dir, exist_ok=True)
    mix_dir = f'datasets/{dataset}/mix'
    mix_filename = f'{mix_dir}/chorale_{chorale}_mix.wav'
    signal = load_audio(mix_filename, n_fft=config.n_fft)

    midi_files, all_pitches = load_midi_files(chorale, dataset)
    W_init = nmf.initialize_components(signal, all_pitches, num_partials=config.num_partials, phi=config.phi)
    H_init, H_sources = initialize_activations(signal, midi_files, all_pitches, config.tol_on, config.tol_off)
    n_components = H_init.shape[0]
    transformer = sklearn.decomposition.NMF(n_components=n_components, solver='mu', init='custom', max_iter=1000)
    W, H = nmf.decompose_custom(signal.X, n_components=n_components, sort=False, transformer=transformer, W=W_init, H=H_init)
    separated_sources = resynthesize_sources(W, H, H_sources, signal, config.tol_on, config.tol_off, config.mask)
    for source_name, separated_source in zip(source_names, separated_sources):
        filename = f'{output_dir}/{source_name}.wav'
        librosa.output.write_wav(filename, separated_source, signal.sr)

def load_audio(filename, sr=22050, n_fft=2048, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    x, audio_sr = librosa.load(filename, sr=None, mono=False)
    assert sr == audio_sr, f'Expected sample rate {sr} but found {audio_sr} in file: {filename}'
    S = librosa.stft(x, n_fft, hop_length)
    X, X_phase = librosa.magphase(S)
    return nmf.Signal(x, sr, S, X, X_phase, n_fft, hop_length)

def resynthesize_sources(W, H, H_source_inits, signal, tol_on, tol_off, use_mask):
    sources = []
    for H_source_init in H_source_inits:
        source_H = H_source_init * H
        if use_mask:
            mask = np.dot(W, source_H) / (np.dot(W, H) + np.finfo(float).eps)
            Y = mask * signal.S
        else:
            Y = np.dot(W, source_H) * signal.X_phase
        reconstructed_signal = librosa.istft(Y, length=len(signal.x))
        sources.append(reconstructed_signal)
    return sources

def load_midi_files(chorale, dataset):
    midi_dir = f'/Users/matan/gdrive/Grad/Thesis/Data/Synthesized/{dataset}/midi'
    files = []
    pitches: List[int] = []
    for source in source_names:
        midi = mido.MidiFile(f'{midi_dir}/chorale_{chorale}_{source}.mid')
        files.append(midi)
        pitches += nmf.get_notes_from_midi(midi)

    return files, sorted(set(pitches))

def initialize_activations(signal, midi_files, all_pitches, tol_on, tol_off):
    H_sources = []
    for midi in midi_files:
        H_sources.append(nmf.initialize_activations(signal, midi, all_pitches, tol_on, tol_off))

    H_init = sum(H_sources)
    np.clip(H_init, 0, 1, out=H_init)
    return H_init, H_sources

test_chorales = [
    '335', '336', '337', '338', '339', '340', '341', '342', '343', '345', '346',
    '349', '350', '351', '352', '354', '355', '356', '357', '358', '359', '360',
    '361', '363', '364', '365', '366', '367', '369', '370', '371'
]

def main(config_name):
    configs_by_name = {c.config_name: c for c in configs}
    for dataset in datasets:
        print(f'Dataset: {dataset}')
        for chorale in test_chorales:
            print(f'\t{chorale}')
            separate(chorale, dataset, configs_by_name[config_name])


configs = [
    NMFConfiguration(
        config_name='A',
        num_partials=None,
        phi=1,
        tol_on=0.2,
        tol_off=1,
        n_fft=2048,
    ),
    NMFConfiguration(
        config_name='B',
        num_partials=None,
        phi=1,
        tol_on=0,
        tol_off=0.2,
        n_fft=2048,
    ),
    NMFConfiguration(
        config_name='C',
        num_partials=None,
        phi=0.4,
        tol_on=0,
        tol_off=0.2,
        n_fft=2048,
    ),
    NMFConfiguration(
        config_name='D',
        num_partials=None,
        phi=0.4,
        tol_on=0,
        tol_off=0.2,
        n_fft=4096,
    ),
]

if __name__ == '__main__':
    main('A')