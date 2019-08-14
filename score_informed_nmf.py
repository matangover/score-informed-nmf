# -*- coding: utf-8 -*-
import librosa
import scipy
import numpy
import ipywidgets
import mido
import IPython.display as ipd
import matplotlib.pyplot as plt
import sklearn
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Iterable
import itertools

def nmf(signal, n_components, **nmf_args):
    transformer = sklearn.decomposition.NMF(n_components=n_components, **nmf_args)
    W, H = librosa.decompose.decompose(signal.X, n_components=n_components, sort=True, transformer=transformer)
    components = get_components(W, H, signal, n_components)
    nmf_display(W, H, signal, components)

def nmf_display(W, H, signal, components):
    display_components(components)

    # Re-create the STFT from all NMF components.
    Y = numpy.dot(W, H) * signal.X_phase

    # Transform the STFT into the time domain.
    print('Reconstructed')
    reconstructed_signal = librosa.istft(Y, length=len(signal.x))
    ipd.display(ipd.Audio(reconstructed_signal, rate=signal.sr))
    
    print('Residual')
    residual = signal.x - reconstructed_signal
    residual[0] = 1 # hack to prevent automatic gain scaling
    ipd.display(ipd.Audio(residual, rate=signal.sr))

def display_components(components):
    all_components = list(components)
    def view_component(component_index):
        if component_index is not None:
            ipd.display(all_components[component_index][1])

    ipywidgets.interact(view_component, component_index=[(c[0], i) for i, c in enumerate(all_components)])

def get_components(W, H, signal, n_components):
    for n in range(n_components):
        # Re-create the STFT of a single NMF component.
        Y = scipy.outer(W[:,n], H[n]) * signal.X_phase

        # Transform the STFT into the time domain.
        y = librosa.istft(Y)

        yield ('Component {}:'.format(n), ipd.Audio(y, rate=signal.sr))

def labelled_boxes(items):
    audio_boxes = []
    for name, audio in items:
        audio_output = ipywidgets.Output()
        audio_output.append_display_data(audio)
        audio_boxes.append(ipywidgets.VBox(
            [ipywidgets.Label(name), audio_output],
            layout=ipywidgets.Layout(align_items="center")
        ))

    return ipywidgets.HBox(audio_boxes, layout=ipywidgets.Layout(flex_flow="row wrap"))

def show_activations(signal, h, component_labels=None, cmap=None):
    plt.figure(figsize=(14, 4))
    cmap = cmap or librosa.display.cmap(h)
    librosa.display.specshow(h, sr=signal.sr, x_axis='time', hop_length=signal.fft_hop_length, cmap=cmap)
    if component_labels:
        tick_locations = range(0, h.shape[0], 4)
        plt.yticks(tick_locations, numpy.array(component_labels)[tick_locations])
    plt.colorbar()
    plt.show()

def show_components(signal, w, spectrogram_ylim=2000):
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(librosa.amplitude_to_db(w), sr=signal.sr, y_axis='linear')
    plt.colorbar()
    plt.ylim((0, spectrogram_ylim))
    plt.show()

def score_informed_nmf(w_init, h_init, signal, pitches):
    n_components = h_init.shape[0]
    transformer = sklearn.decomposition.NMF(n_components=n_components, solver='mu', init='custom', max_iter=1000)
    W, H = decompose_custom(signal.X, n_components=n_components, sort=False, transformer=transformer, W=w_init, H=h_init)
    print(f'Done. Iterations: {transformer.n_iter_}')
    components = get_components_score_informed(W, H, n_components, signal, pitches)
    nmf_display(W, H, signal, components)
    return W, H

def score_informed_nmf_display(signal, pitches, W_init, H_init):
    show_components(signal, W_init)
    show_activations(signal, H_init)
    W, H = score_informed_nmf(W_init, H_init, signal, pitches)
    resynthesize_sources(W, H, pitches, signal)
    show_components(signal, W)
    show_activations(signal, H)

def score_informed_nmf_display_midi(signal, midi, W_init, H_init, tol_on, tol_off, spectrogram_ylim=2000):
    show_components(signal, W_init, spectrogram_ylim=spectrogram_ylim)
    show_activations(signal, H_init)
    W, H = score_informed_nmf(W_init, H_init, signal, get_pitches(midi))
    resynthesize_sources_midi(W, H, midi, signal, tol_on, tol_off)
    show_components(signal, W, spectrogram_ylim=spectrogram_ylim)
    show_activations(signal, H)

def resynthesize_sources_midi(W, H, midi, signal, tol_on, tol_off):
    channel_activations = initialize_activations(signal, midi, get_pitches(midi), tol_on, tol_off, by_channel=True)
    for channel in sorted(channel_activations.keys()):
        channel_H = channel_activations[channel] * H
        Y = numpy.dot(W, channel_H) * signal.X_phase
        print(f'Channel {channel}:')
        reconstructed_signal = librosa.istft(Y, length=len(signal.x))
        ipd.display(ipd.Audio(reconstructed_signal, rate=signal.sr))

        mask = numpy.dot(W, channel_H) / (numpy.dot(W, H) + numpy.finfo(float).eps)
        Y2 = mask * signal.S
        print(f'Channel {channel} (masked):')
        reconstructed_signal2 = librosa.istft(Y2, length=len(signal.x))
        ipd.display(ipd.Audio(reconstructed_signal2, rate=signal.sr))

def score_informed_nmf_with_onset_templates_display(signal, pitches, W_init, H_init):
    show_components(signal, W_init)
    show_activations(signal, H_init)
    W, H = score_informed_nmf_with_onset_templates(W_init, H_init, signal, pitches)
    show_components(signal, W)
    show_activations(signal, H)

def get_notes_from_midi(mid, max_time=None):
    notes = []
    current_time = 0
    for msg in mid:
        if msg.is_meta:
            continue
        if msg.type == 'note_on':
            notes.append(msg.note)
        current_time += msg.time
        if max_time is not None and current_time > max_time:
            break
    return notes

def get_components_score_informed(W, H, n_components, signal, pitches):
    for n in range(n_components):
        # Re-create the STFT of a single NMF component.
        Y = scipy.outer(W[:,n], H[n]) * signal.X_phase

        # Transform the STFT into the time domain.
        y = librosa.istft(Y)
        label = pitches[n]
        if not isinstance(label, str):
            label = librosa.midi_to_note(label)

        yield (label, ipd.Audio(y, rate=signal.sr))

def resynthesize_sources(W, H, pitches, signal):
    lh_max_pitch = librosa.note_to_midi('A3')
    components = numpy.array(pitches)
    
    H1 = numpy.copy(H)
    H2 = numpy.copy(H)
    H1[components > lh_max_pitch, :] = 0
    H2[components <= lh_max_pitch, :] = 0
    Y1 = numpy.dot(W, H1) * signal.X_phase
    Y2 = numpy.dot(W, H2)* signal.X_phase

    print('Left hand:')
    reconstructed_signal1 = librosa.istft(Y1, length=len(signal.x))
    ipd.display(ipd.Audio(reconstructed_signal1, rate=signal.sr))

    print('Right hand:')
    reconstructed_signal2 = librosa.istft(Y2, length=len(signal.x))
    ipd.display(ipd.Audio(reconstructed_signal2, rate=signal.sr))

def initialize_components(signal, pitches, num_partials=15, phi=0.5):
    n_features, _n_samples = signal.S.shape
    fft_freqs = librosa.fft_frequencies(signal.sr, signal.n_fft)
    n_components = len(pitches)
    W_init = numpy.zeros((n_features, n_components))
    phi_below = phi if isinstance(phi, (float, int)) else phi[0]
    phi_above = phi if isinstance(phi, (float, int)) else phi[1]
    for i, pitch in enumerate(pitches):
        if num_partials is None:
            partials: Iterable[int] = itertools.count(start=1)
        else:
            partials = range(1, num_partials + 1)
        for partial in partials:
            min_freq = librosa.midi_to_hz(pitch - phi_below) * partial
            if min_freq > fft_freqs[-1]:
                break
            max_freq = librosa.midi_to_hz(pitch + phi_above)  * partial
            max_freq = min(fft_freqs[-1], max_freq)
            intensity = 1 / (partial**2)
            start_bin = freq_to_bin(min_freq, fft_freqs, round='down')
            end_bin = freq_to_bin(max_freq, fft_freqs, round='up')
            W_init[start_bin:end_bin+1,i] = intensity
    return W_init

def initialize_components_midi(signal, midi, num_partials=15, phi=0.5):
    return initialize_components(signal, get_pitches(midi), num_partials, phi)

def initialize_components_unlimited_partials(signal, pitches, phi=1):
    n_features, _n_samples = signal.S.shape
    n_components = len(pitches)
    W_init = numpy.zeros((n_features, n_components))
    fft_freqs = librosa.fft_frequencies(signal.sr, signal.n_fft)
    for i, pitch in enumerate(pitches):
        #print(i, pitch)
    #    freq = librosa.midi_to_hz(pitch)
        partial = 1
        while True:
            min_freq = librosa.midi_to_hz(pitch - phi) * partial
            max_freq = librosa.midi_to_hz(pitch + phi)  * partial
            max_freq = min(fft_freqs[-1], max_freq)
            intensity = 1 / (partial**2)
            #print('\t%s-%s (%s-%s): %s' % (freq_to_bin(min_freq),freq_to_bin(max_freq), min_freq, max_freq, intensity))
            W_init[freq_to_bin(min_freq, fft_freqs):freq_to_bin(max_freq, fft_freqs),i] = intensity
            if max_freq >= fft_freqs[-1]:
                break
            partial += 1
    return W_init

def freq_to_bin(freq, fft_freqs, round='down'):
    if round == 'down':
        return numpy.searchsorted(fft_freqs, freq, side='right') - 1
    else:
        return numpy.searchsorted(fft_freqs, freq, side='left')

def decompose_custom(S, n_components, transformer, sort=False, W=None, H=None):
    '''
    W : array-like, shape (n_samples, n_components)
        If init=’custom’, it is used as initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        If init=’custom’, it is used as initial guess for the solution.
    '''
    # H and W must be reversed because they are transposed
    activations = transformer.fit_transform(S.T, W=H.T, H=W.T).T
    components = transformer.components_.T

    if sort:
        components, idx = librosa.util.axis_sort(components, index=True)
        activations = activations[idx]

    return components, activations

def initialize_activations_random(signal, pitches):
    _n_features, n_samples = signal.S.shape
    n_components = len(pitches)
    avg = numpy.sqrt(signal.S.mean() / n_components)
    H_init = avg * numpy.random.randn(n_components, n_samples)
    # W = avg * rng.randn(n_samples, n_components)
    H_init = numpy.abs(H_init)
    #np.abs(W, W)
    return H_init

def initialize_activations(signal, mid, pitches, tol_on, tol_off, by_channel=False, max_time=None):
    _n_features, n_samples = signal.S.shape
    n_components = len(pitches)
    H_init: Union[numpy.ndarray, Dict[int, numpy.ndarray]]
    if by_channel:
        H_init = defaultdict(lambda: numpy.zeros((n_components, n_samples)))
    else:
        H_init = numpy.zeros((n_components, n_samples))

    current_time = 0
    on_since = {}
    for msg in mid:
        current_time += msg.time
        if msg.is_meta:
            continue
        if msg.type == 'note_on' and msg.velocity != 0:
            on_since[msg.note] = current_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note not in on_since:
                print(f'note_off without note_on: {msg.note} at time {current_time}')
                continue
            note_on_since = on_since.pop(msg.note)
            component = pitch_to_component(msg.note, pitches)
            start_frame, end_frame = librosa.time_to_frames([note_on_since - tol_on, current_time + tol_off], sr=signal.sr, hop_length=signal.fft_hop_length)
            start_frame = max(start_frame, 0)
            end_frame = min(end_frame, n_samples - 1)
            activations: numpy.ndarray = H_init[msg.channel] if by_channel else H_init # type: ignore
            activations[component, start_frame:end_frame] = 1
        if max_time is not None and current_time > max_time:
           break

    return H_init

def initialize_activations_midi(signal, mid, tol_on, tol_off, by_channel=False):
    return initialize_activations(signal, mid, get_pitches(mid), tol_on, tol_off, by_channel)

def get_pitches(mid):
    notes = get_notes_from_midi(mid)
    return sorted(set(notes))

def pitch_to_component(pitch, pitches):
    return pitches.index(pitch)

def initialize_activations_with_onset_templates(signal, mid, pitches, tol_on, tol_off, max_time=None):
    _n_features, n_samples = signal.S.shape
    H_init = numpy.zeros((len(pitches) * 2, n_samples))
    onset_components_offset = len(pitches)
    current_time = 0
    on_since = {}
    for msg in mid:
        if msg.is_meta:
            continue
        current_time += msg.time
        if msg.type == 'note_on':
            on_since[msg.note] = current_time
            component = pitch_to_component(msg.note, pitches)
            start_frame, end_frame = librosa.time_to_frames([current_time - tol_on, current_time + tol_on], sr=signal.sr, hop_length=signal.fft_hop_length)
            start_frame = max(start_frame, 0)
            end_frame = min(end_frame, n_samples - 1)
            H_init[onset_components_offset + component, start_frame:end_frame] = 1
        elif msg.type == 'note_off':
            note_on_since = on_since.pop(msg.note)
            component = pitch_to_component(msg.note, pitches)
            start_frame, end_frame = librosa.time_to_frames([note_on_since - tol_on, current_time + tol_off], sr=signal.sr, hop_length=signal.fft_hop_length)
            start_frame = max(start_frame, 0)
            end_frame = min(end_frame, n_samples - 1)
            H_init[component, start_frame:end_frame] = 1
        if max_time is not None and current_time > max_time:
            break

    return H_init

def initialize_onset_components_uniform(signal, pitches, value=1):
    n_features, _n_samples = signal.S.shape
    n_components = len(pitches)
    return numpy.full((n_features, n_components), value)

def get_components_score_informed_with_onset_templates(W, H, n_components, signal, pitches):
    for n in range(n_components*2):
        # Re-create the STFT of a single NMF component.
        Y = scipy.outer(W[:,n], H[n]) * signal.X_phase

        # Transform the STFT into the time domain.
        y = librosa.istft(Y)
        if n < n_components:
            label = "note: "
        else:
            n = n - n_components
            label = "onset: "
        label += librosa.midi_to_note(pitches[n])
        yield ('Component {} ({}):'.format(n, label), ipd.Audio(y, rate=signal.sr))

def resynthesize_sources_with_onset_templates(W, H, pitches, signal):
    lh_max_pitch = librosa.note_to_midi('A3')
    components = numpy.array(sorted(pitches) + sorted(pitches))
    
    H1 = numpy.copy(H)
    H2 = numpy.copy(H)
    H1[components > lh_max_pitch, :] = 0
    H2[components <= lh_max_pitch, :] = 0
    Y1 = numpy.dot(W, H1)*signal.X_phase
    Y2 = numpy.dot(W, H2)*signal.X_phase

    print('Left hand:')
    reconstructed_signal1 = librosa.istft(Y1, length=len(signal.x))
    ipd.display(ipd.Audio(reconstructed_signal1, rate=signal.sr))

    print('Right hand:')
    reconstructed_signal2 = librosa.istft(Y2, length=len(signal.x))
    ipd.display(ipd.Audio(reconstructed_signal2, rate=signal.sr))

def score_informed_nmf_with_onset_templates(w_init, h_init, signal, pitches):
    n_components = len(pitches)
    transformer = sklearn.decomposition.NMF(n_components=n_components*2, solver='mu', init='custom')
    W, H = decompose_custom(signal.X, n_components=n_components*2, sort=False, transformer=transformer, W=w_init, H=h_init)
    components = get_components_score_informed_with_onset_templates(W, H, n_components, signal, pitches)
    nmf_display(W, H, signal, components)
    resynthesize_sources_with_onset_templates(W, H, pitches, signal)
    return W, H

def initialize_onset_components_random(signal, pitches):
    n_features, _n_samples = signal.S.shape
    n_components = len(pitches)
    avg = numpy.sqrt(signal.X.mean() / (n_components * 2))
    W_init = avg * numpy.random.randn(n_features, n_components)
    numpy.abs(W_init, out=W_init)
    return W_init

def show_magnitudes(X, sr):
    db = librosa.amplitude_to_db(X)
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.ylim((0, 2000))

def load_audio(filename, sr=22050, n_fft=2048, hop_length=None, display=True):
    if hop_length is None:
        hop_length = n_fft // 4
    x, sr = librosa.load(filename, sr=sr)
    if display:
        ipd.display(ipd.Audio(x, rate=sr))
    S = librosa.stft(x, n_fft, hop_length)
    X, X_phase = librosa.magphase(S)
    if display:
        show_magnitudes(X, sr)
    return Signal(x, sr, S, X, X_phase, n_fft, hop_length)

@dataclass
class Signal:
    x: numpy.ndarray
    sr: int
    S: numpy.ndarray
    X: numpy.ndarray
    X_phase: numpy.ndarray
    n_fft: int
    fft_hop_length: int

def initialize_components_and_activations(score, signal, tol_on=0.1, tol_off=0.1):
    pitches = sorted(set(note[0] for note in score))
    W_init = initialize_components(signal, pitches)
    H_init = initialize_activations_from_score(signal, pitches, score, tol_on, tol_off)
    return NMFInitialization(W_init, H_init, pitches)


def initialize_activations_from_score(signal, pitches, score, tol_on, tol_off):
    n_components = len(pitches)
    _n_features, n_samples = signal.S.shape
    H_init = numpy.zeros((n_components, n_samples))
    for note in score:
        component = pitch_to_component(note[0], pitches)
        start_frame, end_frame = librosa.time_to_frames([note[1] - tol_on, note[2] + tol_off], sr=signal.sr, hop_length=signal.fft_hop_length)
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, n_samples - 1)
        H_init[component, start_frame:end_frame] = 1
    
    return H_init

NMFInitialization = namedtuple('NMFInitialization', ['W_init', 'H_init', 'pitches'])

def initialize_components_random(signal, midi):
    n_features, _n_samples = signal.S.shape
    n_components = len(get_pitches(midi))
    avg = numpy.sqrt(signal.S.mean() / n_components)
    W_init = avg * numpy.random.randn(n_features, n_components)
    W_init = numpy.abs(W_init)
    return W_init

@dataclass
class ScoreInitialization:
    component_groups: List['ComponentGroupInitialization']

@dataclass
class ComponentGroupInitialization:
    activations: 'Initialization'
    components: 'Initialization'

class Initialization:
    def __init__(self):
        raise NotImplementedError()

    def initialize(self, signal, midi):
        self.n_features, self.n_samples = signal.S.shape
        self.n_components = len(get_pitches(midi))
        return self._initialize(signal, midi)
    
    def _initialize(self, signal, midi):
        raise NotImplementedError()

def normalized_random(shape, signal, n_components):
    avg = numpy.sqrt(signal.S.mean() / n_components)
    H_init = avg * numpy.random.randn(*shape)
    H_init = numpy.abs(H_init)
    return H_init

class RandomComponentInitialization(Initialization):
    def __init__(self):
        pass

    def _initialize(self, signal, midi):
        return normalized_random((self.n_features, self.n_components), signal, self.n_components)    

# class RandomActivationInitialization(Initialization):
#     def _initialize(self, signal, midi):
#         return normalized_random((self.n_components, self.n_samples), signal, self.n_components)

@dataclass
class MidiComponentInitialization(Initialization):
    phi: float
    num_partials: int

    def _initialize(self, signal, midi):
        return initialize_components(signal, get_pitches(midi), self.num_partials, self.phi)

@dataclass
class MidiActivationInitialization(Initialization):
    tol_on: float
    tol_off: float

    def _initialize(self, signal, midi):
        by_channel: Dict = initialize_activations_midi(signal, midi, self.tol_on, self.tol_off, True)
        channel_activations = [v for k, v in sorted(by_channel.items(), key=lambda item: item[0])]
        channel_sum = numpy.sum(channel_activations, axis=0)
        total_activations = numpy.clip(channel_sum, a_min=None, a_max=1)
        return total_activations, channel_activations


def score_informed_nmf_display_params(signal, midi, init_params: ScoreInitialization):
    W_init = initialize_components_params(signal, midi, init_params)
    H_init, source_activations = initialize_activations_params(signal, midi, init_params)

    component_labels = [
        f'{librosa.midi_to_note(pitch)} ({component_index + 1})'
        for component_index in range(len(init_params.component_groups))
        for pitch in get_pitches(midi)]

    show_components(signal, W_init, spectrogram_ylim=None)
    show_activations(signal, H_init, component_labels=component_labels)
    W, H = score_informed_nmf(W_init, H_init, signal, component_labels)
    resynthesize_sources_params(W, H, midi, signal, source_activations)
    show_components(signal, W, spectrogram_ylim=None)
    show_activations(signal, H, component_labels=component_labels)
    return W, H

def resynthesize_sources_params(W, H, midi, signal, source_activations: List[numpy.ndarray]) -> None:
    for source_index, source in enumerate(source_activations):
        channel_H = source * H
        Y = numpy.dot(W, channel_H) * signal.X_phase
        print(f'Channel {source_index}:')
        reconstructed_signal = librosa.istft(Y, length=len(signal.x))
        ipd.display(ipd.Audio(reconstructed_signal, rate=signal.sr))

        mask = numpy.dot(W, channel_H) / (numpy.dot(W, H) + numpy.finfo(float).eps)
        Y2 = mask * signal.S
        print(f'Channel {source_index} (masked):')
        reconstructed_signal2 = librosa.istft(Y2, length=len(signal.x))
        ipd.display(ipd.Audio(reconstructed_signal2, rate=signal.sr))

def initialize_activations_params(signal: Signal, mid, init_params: ScoreInitialization, by_channel=False) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
    total_activations = numpy.empty((0, signal.S.shape[1]))
    source_activations: List[numpy.ndarray] = []
    for component_group in init_params.component_groups:
        group_total_activations, group_source_activations = component_group.activations.initialize(signal, mid)
        total_activations = numpy.concatenate((total_activations, group_total_activations), axis=0)
        if not source_activations:
            source_activations = group_source_activations
        else:
            source_activations = [
                numpy.concatenate((existing, current), axis=0)
                for existing, current in zip(source_activations, group_source_activations)
            ]
    
    return total_activations, source_activations

def initialize_components_params(signal, midi, init_params: ScoreInitialization):
    components = numpy.empty((signal.S.shape[0], 0))
    for component_group in init_params.component_groups:
        group_components = component_group.components.initialize(signal, midi)
        components = numpy.concatenate((components, group_components), axis=1)
        
    return components
