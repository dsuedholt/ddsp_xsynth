from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from typing import List, Dict

import torch
from torch import nn
from torch import Tensor


class DDSPXSynthWrapper(WaveformToWaveformBase):
    def __init__(
        self,
        model: nn.Module,
        use_debug_mode: bool = True,
        dataset_name: str = "example",
        author_name: List[str] = None,
    ) -> None:
        super().__init__(model, use_debug_mode)
        if not author_name:
            self.author_name = ["Author Name"]
        else:
            self.author_name = author_name

        self.dataset_name = dataset_name

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_model_name(self) -> str:
        return "ddspXsynth" + self.dataset_name

    def get_model_authors(self) -> List[str]:
        return self.author_name

    def get_model_short_description(self) -> str:
        return f"ddspXsynth model trained on {self.dataset_name} data"

    def get_model_long_description(self) -> str:
        return f"A DDSP timbre transfer model trained on {self.dataset_name} data that performs cross-synthesis between the input signal and the DDSP output."
    
    def get_technical_description(self) -> str:
        return f"A DDSP timbre transfer model trained on {self.dataset_name} data that performs cross-synthesis between the input signal and the DDSP output."

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "DDSP", "cross synthesis", self.dataset_name]

    def get_model_version(self) -> str:
        return "0.1.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(
                name="Pitch Shift",
                description="Apply pitch shift (-24 to +24 semitones)",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="XSynth Harmonics",
                description="How much the input signal is mixed into harmonics",
                default_value=0.0,
            ),
            NeutoneParameter(
                name="XSynth Noise",
                description="How much the input signal is mixed into noise",
                default_value=0.0,
            ),
            NeutoneParameter(name='Reverb Mix', description='Mix of IR reverb', default_value=0.5),
        ]

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        MAX_SHIFT = 24
        pshift = (params["Pitch Shift"] - 0.5) * 2 * MAX_SHIFT
        semishift = torch.round(pshift)
        f0_mult = 2 ** (semishift / 12)

        harm_xsynth = params["XSynth Harmonics"]
        noise_xsynth = params["XSynth Noise"]
        ir_mix = params["Reverb Mix"]
        cond_params = {"harm_xsynth": harm_xsynth, "noise_xsynth": noise_xsynth, "irmix": ir_mix}

        out, data = self.model(x, f0_mult=f0_mult, param=cond_params)
        return out
