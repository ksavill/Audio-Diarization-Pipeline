import numpy as np
import soundfile as sf
from numba import cuda, njit, prange
import math
import os

@njit(parallel=True)
def compute_amplitude(audio_data, frame_size):
    amplitude = np.zeros(audio_data.shape[0] // frame_size, dtype=np.float32)
    for i in prange(amplitude.shape[0]):
        start = i * frame_size
        end = start + frame_size
        if end > audio_data.shape[0]:
            end = audio_data.shape[0]
        frame = audio_data[start:end]
        amplitude[i] = np.max(np.abs(frame))
    return amplitude

@cuda.jit
def detect_silence(amplitude, silence_mask, threshold):
    idx = cuda.grid(1)
    if idx < amplitude.size:
        if amplitude[idx] < threshold:
            silence_mask[idx] = 1
        else:
            silence_mask[idx] = 0

def remove_silence_cuda(input_file="denoised_audio.wav", padding_ms=500, silence_thresh_ratio=0.05,
                        min_silence_len=500, frame_size_ms=10, skip_if_exists: bool = False):
    """
    Removes silent periods from an audio file using CUDA for acceleration and adds padding around speech segments.
    Saves the output file in the 'silent_periods_removed' directory.
    
    Parameters:
    - input_file (str): Path to the input WAV file.
    - padding_ms (int): Padding in milliseconds to add around speech. Default is 500ms.
    - silence_thresh_ratio (float): Ratio of max amplitude to set as silence threshold. Default is 0.05 (5%).
    - min_silence_len (int): Minimum length of silence to be considered for splitting (in ms). Default is 500ms.
    - frame_size_ms (int): Frame size in milliseconds for amplitude calculation. Default is 10ms.
    """
    # Ensure output directory exists
    output_folder = "silent_periods_removed"
    os.makedirs(output_folder, exist_ok=True)

    # Generate output file path in the designated folder
    base_name = os.path.basename(input_file)
    output_path = os.path.join(output_folder, base_name)

    if skip_if_exists and os.path.exists(output_path):
        print(f"Output file already exists at {output_path}. Skipping processing.")
        return output_path

    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    # Ensure the input file is a WAV file
    if not input_file.lower().endswith('.wav'):
        print("Input file must be a WAV file.")
        return

    print(f"Loading audio file '{input_file}'...")
    audio_data, sample_rate = sf.read(input_file)
    
    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) == 2:
        audio_data = audio_data.mean(axis=1)

    print(f"Sample Rate: {sample_rate} Hz")
    total_duration_ms = (len(audio_data) / sample_rate) * 1000
    print(f"Total Duration: {total_duration_ms:.2f} ms")

    frame_size = int(sample_rate * frame_size_ms / 1000)  # Number of samples per frame
    print(f"Frame Size: {frame_size} samples ({frame_size_ms} ms)")

    print("Computing amplitude per frame...")
    amplitude = compute_amplitude(audio_data, frame_size)

    max_amplitude = np.max(amplitude)
    silence_thresh = silence_thresh_ratio * max_amplitude
    print(f"Silence Threshold: {silence_thresh} (Amplitude)")

    # Initialize silence mask
    silence_mask = np.zeros(amplitude.shape, dtype=np.int32)

    threads_per_block = 256
    blocks_per_grid = math.ceil(amplitude.size / threads_per_block)

    # Transfer data to device
    d_amplitude = cuda.to_device(amplitude)
    d_silence_mask = cuda.to_device(silence_mask)
    d_threshold = silence_thresh

    print("Detecting silence using CUDA...")
    detect_silence[blocks_per_grid, threads_per_block](d_amplitude, d_silence_mask, d_threshold)
    silence_mask = d_silence_mask.copy_to_host()

    # Identify silent frames
    silent_frames = silence_mask == 1

    # Convert frames back to time in ms
    frame_duration_ms = frame_size_ms
    silence_duration_ms = frame_duration_ms * 1  # Each frame represents frame_size_ms

    # Find silent segments longer than min_silence_len
    min_silence_frames = math.ceil(min_silence_len / frame_duration_ms)

    silent_indices = np.where(silent_frames)[0]

    # Group silent frames into continuous segments
    silent_segments = []
    if len(silent_indices) > 0:
        current_segment = [silent_indices[0]]
        for idx in silent_indices[1:]:
            if idx == current_segment[-1] + 1:
                current_segment.append(idx)
            else:
                if len(current_segment) >= min_silence_frames:
                    silent_segments.append((current_segment[0], current_segment[-1]))
                current_segment = [idx]
        # Check last segment
        if len(current_segment) >= min_silence_frames:
            silent_segments.append((current_segment[0], current_segment[-1]))

    print(f"Detected {len(silent_segments)} silent segments longer than {min_silence_len} ms.")

    # Determine non-silent segments
    non_silent_segments = []
    prev_end = 0
    for start, end in silent_segments:
        non_silent_segments.append((prev_end, start))
        prev_end = end
    non_silent_segments.append((prev_end, len(amplitude)))

    # Convert frame indices to sample indices
    segments = []
    for start_frame, end_frame in non_silent_segments:
        start_sample = start_frame * frame_size
        end_sample = end_frame * frame_size
        segments.append((start_sample, end_sample))

    # Add padding
    padding_samples = int((padding_ms / 1000) * sample_rate)
    padded_segments = []
    for start, end in segments:
        start_padded = max(start - padding_samples, 0)
        end_padded = min(end + padding_samples, len(audio_data))
        padded_segments.append((start_padded, end_padded))

    # Merge overlapping segments
    merged_segments = []
    if padded_segments:
        current_start, current_end = padded_segments[0]
        for start, end in padded_segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))

    print(f"Merging into {len(merged_segments)} segments after padding.")

    # Extract non-silent audio
    processed_audio = np.array([], dtype=audio_data.dtype)
    for idx, (start, end) in enumerate(merged_segments):
        print(f"Segment {idx+1}: {start} to {end} samples ({(start/sample_rate)*1000:.2f}ms to {(end/sample_rate)*1000:.2f}ms)")
        processed_audio = np.concatenate((processed_audio, audio_data[start:end]))

    print(f"Total Processed Duration: {(len(processed_audio)/sample_rate)*1000:.2f} ms")

    # Save the processed audio
    print(f"Exporting processed audio to '{output_path}'...")
    sf.write(output_path, processed_audio, sample_rate)
    print("Processing complete.")

if __name__ == "__main__":
    input_file = "isolated_vocals.wav"
    padding_ms = 500  # Default padding of half a second

    remove_silence_cuda(input_file, padding_ms=padding_ms)