from cqt_pytorch import CQT
from typing import Optional
from torch.nn import functional as F
from torch import Tensor

class CQT_new(CQT):
    def __init__(
        self,
        num_octaves: int,
        num_bins_per_octave: int,
        sample_rate: int,
        hop_size: Optional[int] = None,
        need_pad: bool = False,
        block_length: Optional[int] = None,
        power_of_2_length: bool = False,
    ):
        if hop_size is None:
            super().__init__(
                num_octaves,
                num_bins_per_octave,
                sample_rate,
                block_length,
                power_of_2_length,
            )
        else:
            if sample_rate % hop_size != 0:
                raise ValueError("sample_rate must be divisible by hop_size, but got sample_rate: {sample_rate}, hop_size: {hop_size}")
            block_length = sample_rate // hop_size
            super().__init__(
                num_octaves,
                num_bins_per_octave,
                sample_rate,
                block_length,
                power_of_2_length,
            )

        self.need_pad = need_pad
        self.hop_size = hop_size

    def encode(self, waveform: Tensor) -> Tensor:
        if self.need_pad:
            self.pad_len = self.block_length - waveform.shape[-1] % self.block_length
            waveform = F.pad(waveform, (0, self.pad_len))
        return super().encode(waveform)

    def decode(self, transform: Tensor) -> Tensor:
        waveform = super().decode(transform)
        if self.need_pad:
            waveform = waveform[:, :, : -self.pad_len]
        return waveform
