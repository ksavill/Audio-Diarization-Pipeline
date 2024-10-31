import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import soundfile as sf
import numpy as np
import os

def remove_background_audio(input_path: str, model_name: str= 'htdemucs', split_duration: int = 300, skip_if_exists: bool = False) -> str:
    """
    Removes background audio from the input audio file and saves the isolated vocals 
    in the 'isolated_vocals' folder with the same file name.
    
    Args:
        input_path (str): Path to the input audio file.
        model_name (str, optional): Name of the Demucs model to use. Defaults to 'htdemucs'.
        split_duration (int, optional): Duration in seconds to split the audio into chunks. Defaults to 300 (5 minutes).
    """
    # Ensure output directory exists
    output_folder = "isolated_vocals"
    os.makedirs(output_folder, exist_ok=True)

    # Generate output file path with the same filename in the designated folder
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, base_name)

    if skip_if_exists and os.path.exists(output_path):
        print(f"Output vocals file already exists at {output_path}. Skipping processing.")
        return output_path

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cpu':
        print("CUDA is not available. For faster performance, consider using a machine with an NVIDIA GPU.")

    # Load the Demucs model
    print(f"Loading Demucs model '{model_name}'...")
    model = get_model(model_name)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Load the audio file using torchaudio
    print(f"Loading audio file '{input_path}'...")
    waveform, sr = torchaudio.load(input_path)
    print(f"Original Sample rate: {sr}, Shape: {waveform.shape}")

    # Resample if necessary
    if sr != model.samplerate:
        print(f"Resampling from {sr} Hz to {model.samplerate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model.samplerate)
        waveform = resampler(waveform)
        sr = model.samplerate
        print(f"Resampled to {sr} Hz.")

    # Ensure waveform is on the correct device
    waveform = waveform.to(device)
    print(f"Audio loaded. Shape: {waveform.shape}")

    # Split the audio into smaller chunks to manage memory
    total_samples = waveform.shape[1]
    split_samples = split_duration * sr
    num_chunks = (total_samples + split_samples - 1) // split_samples  # Ceiling division

    print(f"Splitting audio into {num_chunks} chunks of {split_duration} seconds each...")

    separated_vocals = []

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * split_samples
            end = min((i + 1) * split_samples, total_samples)
            print(f"Processing chunk {i+1}/{num_chunks}: samples {start} to {end}...")
            chunk = waveform[:, start:end]

            # Add batch dimension
            chunk = chunk.unsqueeze(0)  # Shape: [1, channels, samples]

            # Perform source separation
            sources = apply_model(model, chunk, device=device)

            # Identify the index for 'vocals'
            available_sources = model.sources
            print(f"Available sources: {available_sources}")

            try:
                vocals_index = available_sources.index('vocals')
            except ValueError:
                print("Vocals source not found in the available sources. Defaulting to the last source.")
                vocals_index = -1  # Default to the last source

            # Extract the vocals
            vocals_tensor = sources[:, vocals_index, :, :]  # Shape: [1, channels, samples]
            vocals = vocals_tensor.squeeze(0).cpu().numpy()  # Shape: [channels, samples]

            # If multiple channels, convert to mono by averaging
            if vocals.ndim > 1:
                vocals = np.mean(vocals, axis=0)  # Shape: [samples]

            separated_vocals.append(vocals)

    # Concatenate all chunks
    print("Concatenating all separated vocals chunks...")
    final_vocals = np.concatenate(separated_vocals)
    
    # Save the vocals to the output path using soundfile
    print(f"Saving isolated vocals to '{output_path}'...")
    sf.write(output_path, final_vocals, sr)
    print("Vocals saved successfully.")
    return output_path

if __name__ == "__main__":
    input_audio = "noisy_audio.wav"

    # Check if input file exists
    if not os.path.isfile(input_audio):
        print(f"Input file '{input_audio}' does not exist. Please provide a valid file path.")
    else:
        remove_background_audio(input_audio)