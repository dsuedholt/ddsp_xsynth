import os, logging, argparse
from pathlib import Path

import torch

from diffsynth.model import EstimatorSynth
from diffsynth.modules.generators import FilteredNoise, Harmonic
from diffsynth.modules.reverb import IRReverb
from diffsynth.synthesizer import Synthesizer
from diffsynth.stream import StreamIRReverb

from realtimeDDSP.diffsynth.stream import replace_modules
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)
from neutone_sdk.utils import save_neutone_model
import torchaudio

from neutone_wrapper import DDSPXSynthWrapper
from xsynth import DDSPXSynth, XSynthStreamHarmonic, XSynthStreamFilteredNoise


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

def get_stream_synth(synth):
    new_ps = []
    new_cs = []
    conditioned = synth.conditioned_params
    for proc, conn in zip(synth.processors, synth.connections):
        if isinstance(proc, Harmonic):
            # Replace with streamable version of harmonic synthesizer
            new_ps.append(XSynthStreamHarmonic(proc.sample_rate, proc.normalize_below_nyquist, proc.name, proc.n_harmonics, proc.freq_range))
            conn_harm = dict(conn)
            conn_harm['harm_xsynth'] = 'harm_xsynth'
            conn_harm['input_distribution'] = 'input_distribution'
            new_cs.append(conn_harm)
            conditioned.extend(['harm_xsynth', 'input_distribution'])
        elif isinstance(proc, FilteredNoise):
            # Replace with streamable version of noise synthesizer
            new_ps.append(XSynthStreamFilteredNoise(proc.filter_size, proc.name, proc.amplitude))
            conn_noise = dict(conn)
            conn_noise['noise_xsynth'] = 'noise_xsynth'
            conn_noise['input_noise'] = 'input_noise'
            new_cs.append(conn_noise)
            conditioned.extend(['noise_xsynth', 'input_noise'])
        elif isinstance(proc, IRReverb):
            # Replace with streamable version of ir reverb
            new_ps.append(StreamIRReverb(proc.ir, proc.name))
            conn_mix = dict(conn)
            # this version has a parameter for adjusting reverb mix
            conn_mix['mix'] = 'irmix'
            new_cs.append(conn_mix)
            conditioned.append('irmix')
        else:
            new_ps.append(proc)
            new_cs.append(conn)
    synth.processors = torch.nn.ModuleList(new_ps)
    # make new synth
    dag = [(proc, conn) for proc, conn in zip(new_ps, new_cs)]
    new_synth = Synthesizer(dag, conditioned=conditioned)
    return new_synth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt',             type=str,   help='')
    parser.add_argument('output',           type=str,   help='model output name')
    parser.add_argument('--folder',   default='./exports', help='output folder')
    parser.add_argument('--sounds',   nargs='*', type=str, default=None, help='directory of sounds to use as example input.')
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    model = EstimatorSynth.load_from_checkpoint(args.ckpt)
    replace_modules(model.estimator)

    replace_weight_from = 'shakuhachi.nm'
    state_dict = torch.jit.load(replace_weight_from).w2w_base.model.estimator.state_dict()
    model.estimator.load_state_dict(state_dict)

    # get streamable hpnir synth with mix parameters
    model.synth = get_stream_synth(model.synth)
    stream_model = DDSPXSynth(model.estimator, model.synth, 48000, hop_size=960)
    dummy = torch.zeros(1, 2048)
    _ = stream_model(dummy, torch.ones(1), {'harm_xsynth': torch.ones(1), 'noise_xsynth': torch.ones(1), 'irmix': torch.ones(1)})
    wrapper = DDSPXSynthWrapper(stream_model)

    soundpairs = []
    if args.sounds is not None:
        sounds = args.sounds
    else:
        sounds = ['realtimeDDSP/data/413204-mono.mp3', 'realtimeDDSP/data/339357-mono.mp3', 'realtimeDDSP/data/test_lead_mono.mp3']
    for sound in sounds:
        wave, sr = torchaudio.load(sound)
        input_sample = AudioSample(wave, sr)
        rendered_sample = render_audio_sample(wrapper, input_sample)
        soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=True, submission=True, audio_sample_pairs=soundpairs
    )