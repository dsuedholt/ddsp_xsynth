import torch
import torch.nn.functional as F
import torchaudio
from typing import Dict, Tuple

from diffsynth.stream import CachedStreamEstimatorFLSynth, StreamHarmonic, StreamFilteredNoise

from diffsynth.f0 import yin_frame, FMIN, FMAX
from diffsynth.spectral import spec_loudness, spectrogram

import diffsynth.util as util
import xsynth_utils

class XSynthStreamFilteredNoise(StreamFilteredNoise):
    def __init__(self, filter_size=257, name='noise', amplitude=1.0, batch_size=1):
        super().__init__(filter_size, name, amplitude, batch_size)
        self.param_sizes['input_noise'] = self.filter_size // 2 + 1
        self.param_range['input_noise'] = (0.0, 1.0)
        self.param_types['input_noise'] = 'raw'
        self.param_sizes['noise_xsynth'] = 1
        self.param_range['noise_xsynth'] = (0.0, 1.0)
        self.param_types['noise_xsynth'] = "raw"

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """generate Gaussian white noise through FIRfilter
        Args:
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        freq_response = params['freq_response']
        batch_size = freq_response.shape[0]
        # noise
        audio = (torch.rand(batch_size, n_samples)*2.0-1.0).to(freq_response.device) * self.amplitude
        
        xsynth = params['noise_xsynth']
        freq_response = freq_response * (1 - xsynth) + params['input_noise'] * 4 * xsynth
        
        filtered = util.fir_filter(audio, freq_response, self.filter_size, padding='valid')
        output = filtered[..., :n_samples]
        cache = F.pad(self.cache, pad=(0, n_samples-self.cache.shape[-1]))
        output = output + cache
        self.cache = filtered[..., n_samples:]
        return output

class XSynthStreamHarmonic(StreamHarmonic):
    def __init__(
        self,
        sample_rate: int = 48000,
        normalize_below_nyquist: bool = True,
        name: str = "harmonic",
        n_harmonics: int = 256,
        freq_range: Tuple[float, float] = ...,
        batch_size: int = 1,
    ):
        super().__init__(
            sample_rate,
            normalize_below_nyquist,
            name,
            n_harmonics,
            freq_range,
            batch_size,
        )
        self.param_sizes["harm_xsynth"] = 1
        self.param_range["harm_xsynth"] = (0.0, 1.0)
        self.param_types["harm_xsynth"] = "raw"
        self.param_sizes["input_distribution"] = n_harmonics
        self.param_range["input_distribution"] = (0.0, 1.0)
        self.param_types["input_distribution"] = "raw"

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        input_harmonics = params["input_distribution"]
        amplitudes = params["amplitudes"]
        harmonic_distribution = params["harmonic_distribution"]
        f0_hz = params["f0_hz"]
        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            harmonic_frequencies = util.get_harmonic_frequencies(
                f0_hz, self.n_harmonics
            )
            harmonic_distribution = util.remove_above_nyquist(
                harmonic_frequencies, harmonic_distribution, self.sample_rate
            )

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution, dim=-1, keepdim=True)
        input_harmonics /= torch.sum(input_harmonics, dim=-1, keepdim=True)

        xsynth = params["harm_xsynth"]
        harmonic_distribution = (
            harmonic_distribution * (1 - xsynth) + input_harmonics * xsynth
        )

        harmonic_amplitudes = amplitudes * harmonic_distribution
        # interpolate with previous params
        harmonic_frequencies = util.get_harmonic_frequencies(f0_hz, self.n_harmonics)
        harmonic_freqs = torch.cat([self.prev_freqs, harmonic_frequencies], dim=1)
        frequency_envelopes = util.resample_frames(harmonic_freqs, n_samples)
        harmonic_amps = torch.cat([self.prev_harm, harmonic_amplitudes], dim=1)
        amplitude_envelopes = util.resample_frames(harmonic_amps, n_samples)
        audio, last_phase = util.oscillator_bank_stream(
            frequency_envelopes,
            amplitude_envelopes,
            sample_rate=self.sample_rate,
            init_phase=self.phase,
        )
        self.phase = last_phase
        self.prev_harm = harmonic_amplitudes[:, -1:]
        self.prev_freqs = harmonic_frequencies[:, -1:]
        return audio


class DDSPXSynth(CachedStreamEstimatorFLSynth):
    def __init__(
        self, estimator, synth, sample_rate, hop_size=960, pitch_min=50., pitch_max=2000.
    ):
        super().__init__(estimator, synth, sample_rate, hop_size, pitch_min, pitch_max)
        self.hann_win = torch.hann_window(2048, periodic=False)
        self.prev_harm_freqs = torch.zeros(256)

    def forward(
        self, audio: torch.Tensor, f0_mult: torch.Tensor, param: Dict[str, torch.Tensor]
    ):
        with torch.no_grad():
            orig_len = audio.shape[-1]
            orig_audio = audio
            # input cache
            audio = torch.cat([self.input_cache.to(audio.device), audio], dim=-1)
            windows = util.slice_windows(
                audio, self.window_size, self.hop_size, pad=False
            )

            self.offset = self.hop_size - ((orig_len - self.offset) % self.hop_size)
            self.input_cache = audio[:, -(self.window_size - self.offset) :]

            f0 = yin_frame(windows, self.sample_rate, self.pitch_min, self.pitch_max)
            # loudness
            comp_spec = torch.fft.rfft(windows, dim=-1)
            loudness = spec_loudness(comp_spec, self.a_weighting)

            if f0[:, 0] == 0:
                # use previous f0 if noisy
                f0[:, 0] = self.prev_f0
                # also assume silent if noisy
                # loudness[:, 0] = 0
            for i in range(1, f0.shape[1]):
                if f0[:, i] == 0:
                    f0[:, i] = f0[:, i - 1]
                    # loudness[:, i] = 0

            self.prev_f0 = f0[:, -1]
            # estimator
            f0 = f0_mult * f0
            x = {
                "f0": f0[:, :, None],
                "loud": loudness[:, :, None],
            }  # batch=1, n_frames=windows.shape[1], 1
            x.update(param)
            est_param = self.estimator(x)

            # get input distribution
            # for simplicity use same distribution for every timestep corresponding to
            # current input buffer
            input_spec = torch.abs(torch.fft.rfft(
                torch.squeeze(orig_audio) * self.hann_win,
                norm="forward"
            ))
            input_spec_db = 10 * torch.log10(input_spec + 1e-10)

            peak_locs = xsynth_utils.detect_peaks(input_spec_db)
            ipeak_locs, ipeak_mags = xsynth_utils.interpolate_peaks(
                input_spec_db, peak_locs
            )
            ipeak_freqs = self.sample_rate * ipeak_locs / 2048

            self.prev_harm_freqs, harm_mags = xsynth_utils.detect_harmonics(
                ipeak_freqs,
                ipeak_mags,
                self.prev_f0[-1],
                256,
                self.prev_harm_freqs,
                self.sample_rate,
            )

            ## convert back from dB to amplitude values
            amp_env = 10 ** (harm_mags / 10)
            x["input_distribution"] = amp_env

            # generate audio from harmonics
            amp_env = amp_env.repeat(orig_audio.numel(), 2)
            freq_env = self.prev_harm_freqs.repeat(orig_audio.numel(), 2)
            input_harm_audio = util.oscillator_bank(freq_env, amp_env, self.sample_rate)
            
            # spectrum of residual noise
            res_spec = torch.abs(torch.fft.rfft(
                torch.squeeze(orig_audio - input_harm_audio) * self.hann_win,
                norm="forward"
            ))
            
            harmonicity = (torch.sum(input_harm_audio ** 2) + 1e-10) / (torch.sum(orig_audio ** 2) + 1e-10)
            
            #res_spec = torchaudio.functional.resample(res_spec, 1025, 512 // 2 + 1, lowpass_filter_width=128)
            res_spec = torchaudio.functional.resample(input_spec, 1025, 512 // 2 + 1, lowpass_filter_width=2)
            x["input_noise"] = res_spec

            params_dict = self.synth.fill_params(est_param, x)
            render_length = (
                windows.shape[1] * self.hop_size
            )  # last_of_prev_frame<->0th window<-...->last window

            resyn_audio, outputs = self.synth(params_dict, render_length)
            # output cache (delay)
            resyn_audio = torch.cat(
                [self.output_cache.to(audio.device), resyn_audio], dim=-1
            )
            if resyn_audio.shape[-1] > orig_len:
                self.output_cache = resyn_audio[:, orig_len:]
                resyn_audio = resyn_audio[:, :orig_len]
            return resyn_audio, (loudness, f0)
