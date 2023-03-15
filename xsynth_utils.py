import sms_utils
import torch


def detect_harmonics(spec_db, f0, n_harmonics: int, prev_harm_freqs, sample_rate: int):
    peak_locs = sms_utils.detect_peaks(spec_db)
    ipeak_locs, ipeak_mags = sms_utils.interpolate_peaks(spec_db, peak_locs)
    nfft = (spec_db.numel() - 1) * 2
    ipeak_freqs = sample_rate * ipeak_locs / nfft

    return sms_utils.detect_harmonics(
        ipeak_freqs,
        ipeak_mags,
        f0,
        n_harmonics,
        prev_harm_freqs,
        sample_rate,
    )


def interpolate_harmonics(harm_freqs, harm_mags, f0_synth, sample_rate: int, formant):
    """
    Sample a given harmonic envelope at harmonics of a new fundamental frequency.
    This allows for timbre-preserving (formant-preserving) pitch shifts.
    Optional formant shift is applied by squashing or stretching the sampling intervals.

    Args:
        harm_freqs: frequencies of the detected harmonics in the input envelope
        harm_mags: magnitudes of the detected harmonics in the input envelope
        f0_synth: new fundamental frequency to sample the envelope at
        sample_rate: sample rate of the audio
        formant: formant shift factor (0.5 = no shift, 0 = full shift down, 1 = full shift up)

    Returns:
        synth_harm_mags: harmonic envelope to be used for additive synthesis
    """

    n_harmonics = harm_freqs.numel()

    present_harmonics = harm_freqs > 0

    if not present_harmonics.any():
        return torch.ones(n_harmonics) * -100

    harm_freqs = harm_freqs[present_harmonics]
    harm_mags = harm_mags[present_harmonics]

    # make sure harm_freqs are sorted, adjust harm_mags accordingly
    sort_idx = torch.argsort(harm_freqs)
    harm_freqs = harm_freqs[sort_idx]
    harm_mags = harm_mags[sort_idx]

    # insert a "pseudo-harmonic" at 0 Hz with magnitude -100 dB (silence) to make interpolation easier
    harm_freqs = torch.cat([torch.zeros(1), harm_freqs])
    harm_mags = torch.cat([torch.ones(1) * -100, harm_mags])

    formant_shift = 2 ** -(formant * 2 - 1)
    synth_harm_mags = torch.ones(n_harmonics) * -100
    synth_harm_freqs = f0_synth * torch.arange(1, n_harmonics + 1) * formant_shift

    # for each frequency in synth_harm_freqs, find the closest (higher) harmonic in harm_freqs
    freq_idxs = torch.searchsorted(harm_freqs, synth_harm_freqs, right=True)

    # discard frequencies that are higher than the highest harmonic in harm_freqs or higher than nyquist
    valid_freqs = (freq_idxs < harm_freqs.numel()) & (
        synth_harm_freqs < sample_rate / 2
    )
    valid_idxs = freq_idxs[valid_freqs]

    # sample harm_mags with linear interpolation
    alphas = (synth_harm_freqs[valid_freqs] - harm_freqs[valid_idxs - 1]) / (
        harm_freqs[valid_idxs] - harm_freqs[valid_idxs - 1]
    )
    synth_harm_mags[valid_freqs] = harm_mags[valid_idxs - 1] + alphas * (
        harm_mags[valid_idxs] - harm_mags[valid_idxs - 1]
    )

    return synth_harm_mags
