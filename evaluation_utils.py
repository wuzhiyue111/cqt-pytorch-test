import numpy as np
import torch
import torchaudio

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio



def calculate_si_snr(gt_audio: torch.Tensor, rec_audio: torch.Tensor, eps=1e-8):
    if gt_audio.dim() > 2:
        gt_audio_cal = gt_audio.clone()
        rec_audio_cal = rec_audio.clone()
        gt_audio_cal = gt_audio_cal.squeeze(1)
        rec_audio_cal = rec_audio_cal.squeeze(1)

    si_snr = scale_invariant_signal_noise_ratio(rec_audio_cal, gt_audio_cal)

    return si_snr


def calculate_stoi(rec_audio: torch.Tensor, gt_audio: torch.Tensor, sample_rate=24000):
    stoi = ShortTimeObjectiveIntelligibility(sample_rate).to(gt_audio.device)
    return stoi(rec_audio, gt_audio).item()


def calculate_pesq(rec_audio: torch.Tensor, gt_audio: torch.Tensor, sample_rate=24000):
    pesq = PerceptualEvaluationSpeechQuality(16000, "wb")

    # PESQ要求采样率为16k或8k
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
        gt_audio = resampler(gt_audio)
        rec_audio = resampler(rec_audio)

    gt_audio_cal = gt_audio.clone()
    rec_audio_cal = rec_audio.clone()
    assert gt_audio_cal.shape == rec_audio_cal.shape

    if gt_audio_cal.dim() == 2:
        gt_audio_cal = gt_audio_cal.view(-1)
        rec_audio_cal = rec_audio_cal.view(-1)
        return pesq(rec_audio_cal, gt_audio_cal)

    elif gt_audio_cal.dim() == 3:
        gt_audio_cal = gt_audio_cal.view(gt_audio_cal.shape[0], -1)
        rec_audio_cal = rec_audio_cal.view(rec_audio_cal.shape[0], -1)
        pesq_list = []
        for i in range(gt_audio_cal.shape[0]):
            try:
                pesq_list.append(pesq(rec_audio_cal[i], gt_audio_cal[i]))
            except Exception as e:
                print(f"gt_audio.shape = {gt_audio_cal.shape}, rec_audio.shape = {rec_audio_cal.shape}, error = {e}")
        return np.mean(np.array(pesq_list))

    elif gt_audio_cal.dim() == 1:
        return pesq(rec_audio_cal, gt_audio_cal)

    else:
        raise ValueError("gt_audio dim must be 1, 2 or 3")