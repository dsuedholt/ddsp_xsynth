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
        author_name: str = "Author Name",
    ) -> None:
        super().__init__(model, use_debug_mode)
        self.author_name = [author_name]

        self.dataset_name = dataset_name

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [self.model.sample_rate]

    def get_native_buffer_sizes(self) -> List[int]:
        return [self.model.window_size]

    def get_model_name(self) -> str:
        return "ddspX" + self.dataset_name

    def get_model_authors(self) -> List[str]:
        return self.author_name

    def get_model_short_description(self) -> str:
        return f"DDSP cross-synthesis model trained on {self.dataset_name} data"

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

    def get_citation(self) -> str:
        return """
        Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP: Differentiable Digital Signal Processing. ICLR.
        """

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(
                name="Pitch Shift",
                description="Apply pitch shift (-24 to +24 semitones)",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="Cross-Synth",
                description="How much the input signal is mixed into harmonics",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="Noise",
                description="Volume of the filter noise",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="Formant", description="Formant Shift", default_value=0.5
            ),
        ]

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        MAX_SHIFT = 24
        pshift = (params["Pitch Shift"] - 0.5) * 2 * MAX_SHIFT
        semishift = torch.round(pshift)
        f0_mult = 2 ** (semishift / 12)

        harm_xsynth = params["Cross-Synth"]
        noise_xsynth = params["Noise"]
        formant = params["Formant"]
        cond_params = {
            "harm_xsynth": harm_xsynth,
            "harmmix": torch.ones(1) * 0.5,
            "noisemix": noise_xsynth,
            "formant": formant,
        }

        out, data = self.model(x, f0_mult=f0_mult, param=cond_params)
        return out
