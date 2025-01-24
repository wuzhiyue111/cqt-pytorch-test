from cqt_pytorch import CQT_new
from evaluation_utils import (
    calculate_pesq,
    calculate_si_snr,
    calculate_stoi,
)
import torchaudio
from torch.nn import functional as F
from time import time

def read_audio(audio_file, time_length=10, mono = True):
    """
        time_length: how long the audio to read, in seconds
    """
    audio, sr = torchaudio.load(audio_file)
    if time_length * sr > audio.shape[-1]:
        audio = audio[:, :sr * time_length]

    if mono:
        if audio.shape[0] > 1:
            audio = audio.mean(0).unsqueeze(0)
    audio = audio.unsqueeze(0)
    return audio, sr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        default="/home/wzy/projects/cqt-pytorch-test/cqt_pytorch/test.wav",
    )
    parser.add_argument(
        "--test_audio_length",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    audio_file = args.audio_file
    root_dir = audio_file.rsplit("/", 1)[0]
    audio, sr = read_audio(audio_file, time_length=args.test_audio_length) # 事例是48000 sample_rate
    transform_hop_size_300 = CQT_new(
        sample_rate=sr,
        num_octaves=8,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=300,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )

    transform_hop_size_600 = CQT_new(
        sample_rate=sr,
        num_octaves=8,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=600,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )

    transform_hop_size_1200 = CQT_new(
        sample_rate=sr,
        num_octaves=8,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1200,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )

    transform_hop_size_2400 = CQT_new(
        sample_rate=sr,
        num_octaves=8,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=2400,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )

    transform_list = [transform_hop_size_300, transform_hop_size_600, transform_hop_size_1200, transform_hop_size_2400]
    gt_audio_list = [audio, audio, audio, audio]
    print("-" * 100)
    print("eval_hop_size_param")
    # evaluate
    for i, (transform, gt_audio) in enumerate(zip(transform_list, gt_audio_list)):
        start_time = time()
        rec_audio = transform.decode(transform.encode(gt_audio))
        time_cost = time() - start_time
        print(f"audio_length: {args.test_audio_length} seconds, decode_time: {time_cost} seconds")
        pesq = calculate_pesq(rec_audio=rec_audio, gt_audio=gt_audio)
        si_snr = calculate_si_snr(rec_audio=rec_audio, gt_audio=gt_audio)
        stoi = calculate_stoi(rec_audio=rec_audio, gt_audio=gt_audio)
        
        print(
            f"eval_hop_size_param: {transform.hop_size}, pesq: {pesq}, si_snr: {si_snr}, stoi: {stoi}, octaves: 8, bins_per_octave: 24"
        )


    print("-" * 100)
    print("eval_octaves_param")
    transform_octaves_8 = CQT_new(
        sample_rate=sr,
        num_octaves=8,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_octaves_9 = CQT_new(
        sample_rate=sr,
        num_octaves=9,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_octaves_10 = CQT_new(
        sample_rate=sr,
        num_octaves=10,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_octaves_11 = CQT_new(
        sample_rate=sr,
        num_octaves=11,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_octaves_12 = CQT_new(
        sample_rate=sr,
        num_octaves=12,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_list = [transform_octaves_8, transform_octaves_9, transform_octaves_10, transform_octaves_11, transform_octaves_12]
    for i, (transform, gt_audio) in enumerate(zip(transform_list, gt_audio_list)):
        octaves_num = i + 8
        start_time = time()
        rec_audio = transform.decode(transform.encode(gt_audio))
        time_cost = time() - start_time
        print(f"audio_length: {args.test_audio_length}, decode_time: {time_cost}")
        pesq = calculate_pesq(rec_audio=rec_audio, gt_audio=gt_audio)
        si_snr = calculate_si_snr(rec_audio=rec_audio, gt_audio=gt_audio)
        stoi = calculate_stoi(rec_audio=rec_audio, gt_audio=gt_audio)
        
        print(
            f"eval_octaves_param: {octaves_num}, pesq: {pesq}, si_snr: {si_snr}, stoi: {stoi}, hop_size: 1024, bins_per_octave: 24"
        )
    
    print("-" * 100)
    print("eval_bins_per_octave_param")
    transform_bins_per_octave_24 = CQT_new(
        sample_rate=sr,
        num_octaves=10,  # 多少个8度
        num_bins_per_octave=24,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_bins_per_octave_48 = CQT_new(
        sample_rate=sr,
        num_octaves=10,  # 多少个8度
        num_bins_per_octave=48,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_bins_per_octave_96 = CQT_new(
        sample_rate=sr,
        num_octaves=10,  # 多少个8度
        num_bins_per_octave=96,  # 每个8度多少个bin
        hop_size=1024,  # hop size
        power_of_2_length=True,
        need_pad=True,
    )
    
    transform_list = [transform_bins_per_octave_24, transform_bins_per_octave_48, transform_bins_per_octave_96]
    for i, (transform, gt_audio) in enumerate(zip(transform_list, gt_audio_list)):
        bins_per_octave = i * 24 + 24
        start_time = time()
        rec_audio = transform.decode(transform.encode(gt_audio))
        time_cost = time() - start_time
        print(f"audio_length: {args.test_audio_length}, decode_time: {time_cost}")
        pesq = calculate_pesq(rec_audio=rec_audio, gt_audio=gt_audio)
        si_snr = calculate_si_snr(rec_audio=rec_audio, gt_audio=gt_audio)
        stoi = calculate_stoi(rec_audio=rec_audio, gt_audio=gt_audio)
        
        print(
            f"eval_bins_per_octave_param: {bins_per_octave}, pesq: {pesq}, si_snr: {si_snr}, stoi: {stoi}, hop_size: 1024, octaves: 10"
        )
        
    print("-" * 100)
    print("end")
