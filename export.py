import os, logging, argparse, sys
from pathlib import Path

import torch

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtimeDDSP")
)

from diffsynth.model import EstimatorSynth
from diffsynth.modules.generators import FilteredNoise, Harmonic
from diffsynth.modules.reverb import IRReverb
from diffsynth.synthesizer import Synthesizer
from diffsynth.stream import StreamIRReverb, StreamFilteredNoise
from diffsynth.processor import Mix

from realtimeDDSP.diffsynth.stream import replace_modules
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)
from neutone_sdk.utils import save_neutone_model
import torchaudio

from neutone_wrapper import DDSPXSynthWrapper
from xsynth import DDSPXSynth, XSynthStreamHarmonic


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
            new_ps.append(
                XSynthStreamHarmonic(
                    proc.sample_rate,
                    proc.normalize_below_nyquist,
                    proc.name,
                    proc.n_harmonics,
                    proc.freq_range,
                )
            )
            conn_harm = dict(conn)
            conn_harm["harm_xsynth"] = "harm_xsynth"
            conn_harm["input_distribution"] = "input_distribution"
            new_cs.append(conn_harm)
            conditioned.extend(["harm_xsynth", "input_distribution"])
        elif isinstance(proc, FilteredNoise):
            # Replace with streamable version of noise synthesizer
            new_ps.append(
                StreamFilteredNoise(proc.filter_size, proc.name, proc.amplitude)
            )
            new_cs.append(conn)
        elif isinstance(proc, IRReverb):
            # remove learned reverb
            pass
        elif proc.name == "add":
            # Replace add module with mix module for adjusting harm/noise
            new_ps.append(Mix(proc.name))
            conn_mix = dict(conn)
            conn_mix["mix_a"] = "harmmix"
            conn_mix["mix_b"] = "noisemix"
            new_cs.append(conn_mix)
            conditioned.extend(["harmmix", "noisemix"])
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
    parser.add_argument("ckpt", type=str, help="")
    parser.add_argument("output", type=str, help="model output name")
    parser.add_argument("--folder", default="./exports", help="output folder")
    parser.add_argument("--dataset_name", type=str, default="example")
    parser.add_argument("--author_name", type=str, default="Author Name")
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    model = EstimatorSynth.load_from_checkpoint(args.ckpt)
    replace_modules(model.estimator)

    # get streamable hpnir synth with mix parameters
    model.synth = get_stream_synth(model.synth)
    stream_model = DDSPXSynth(model.estimator, model.synth, 48000, hop_size=960)
    dummy = torch.zeros(1, 2048)
    _ = stream_model(
        dummy,
        torch.ones(1),
        {
            "harm_xsynth": torch.ones(1),
            "harmmix": torch.ones(1),
            "noisemix": torch.ones(1),
            "formant": torch.ones(1),
        },
    )
    wrapper = DDSPXSynthWrapper(
        stream_model, author_name=args.author_name, dataset_name=args.dataset_name
    )

    wave, sr = torchaudio.load("example_sound.wav")
    input_sample = AudioSample(wave, sr)
    params = (
        torch.tensor([0.5, 0.75, 0.25, 0.5]).repeat((stream_model.window_size, 1)).T
    )
    rendered_sample = render_audio_sample(wrapper, input_sample, params=params)
    soundpairs = [AudioSamplePair(input_sample, rendered_sample)]

    save_neutone_model(
        wrapper,
        root_dir,
        freeze=False,
        dump_samples=True,
        submission=True,
        audio_sample_pairs=soundpairs,
    )
