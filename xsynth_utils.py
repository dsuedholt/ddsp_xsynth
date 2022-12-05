import torch

# mostly copied code from
# https://github.com/MTG/sms-tools/blob/master/software/models/
# replaced np calls with torch calls
# removed phase calculations, not needed for this application


def detect_peaks(mX, t=torch.tensor(-95., dtype=torch.float32)):
    """
    Detect spectral peak locations
    mX: magnitude spectrum (dB), t: threshold
    returns ploc: peak locations
    """

    z = torch.zeros_like(mX[1:-1])

    thresh = torch.where(
        torch.greater(mX[1:-1], t), mX[1:-1], z
    )  # locations above threshold
    next_minor = torch.where(
        mX[1:-1] > mX[2:], mX[1:-1], z
    )  # locations higher than the next one
    prev_minor = torch.where(
        mX[1:-1] > mX[:-2], mX[1:-1], z
    )  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[:, 0] + 1  # add 1 to compensate for previous steps
    return ploc


def interpolate_peaks(mX, ploc):
    """
    Interpolate peak values using parabolic interpolation
    mX: magnitude spectrum, ploc: locations of peaks
    returns iploc, ipmag: interpolated peak location and magnitude values
    """

    val = mX[ploc]  # magnitude of peak bin
    lval = mX[ploc - 1]  # magnitude of bin at left
    rval = mX[ploc + 1]  # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)  # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # magnitude of peaks
    return iploc, ipmag


def detect_harmonics(pfreq: torch.Tensor, pmag: torch.Tensor, f0: torch.Tensor, nH: int, hfreqp: torch.Tensor, fs: int, harmDevSlope=torch.tensor(0.01, dtype=torch.float32)):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag: peak frequencies and magnitude
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns hfreq, hmag: harmonic frequencies and magnitudes
    """

    if f0 <= 0:  # if no f0 return no harmonics
        return torch.zeros(nH), torch.zeros(nH)
    hfreq = torch.zeros(nH)  # initialize harmonic frequencies
    hmag = torch.zeros(nH) - 100  # initialize harmonic magnitudes
    hf = f0 * torch.arange(1, nH + 1)  # initialize harmonic frequencies
    hi = 0  # initialize harmonic index
    if hfreqp.nonzero().numel() == 0:  # if no incomming harmonic tracks initialize to harmonic series
        hfreqp = hf
    while pfreq.numel() > 0 and (f0 > 0) and (hi < nH) and (hf[hi] < fs / 2):  # find harmonic peaks
        pei = torch.argmin(abs(pfreq - hf[hi]))  # closest peak
        dev1 = abs(pfreq[pei] - hf[hi])  # deviation from perfect harmonic
        dev2 = (
            abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi] > 0 else torch.tensor(fs, dtype=torch.float32)
        )  # deviation from previous frame
        threshold = f0 / 3 + harmDevSlope * pfreq[pei]
        if (dev1 < threshold) or (
            dev2 < threshold
        ):  # accept peak if deviation is small
            hfreq[hi] = pfreq[pei]  # harmonic frequencies
            hmag[hi] = pmag[pei]  # harmonic magnitudes
        hi += 1  # increase harmonic index
    return hfreq, hmag
