import os
import ffmpeg

def convert_to_wav(input_video_path: str, skip_if_exists: bool = False) -> str:
    output_folder = "wav_conversions"
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.wav")

    if skip_if_exists and os.path.exists(output_path):
        print(f"Output WAV file already exists at {output_path}. Skipping conversion.")
        return output_path
    
    try:
        (
            ffmpeg
            .input(input_video_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ar='44100', ac=2)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio extracted successfully to {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print("Error extracting audio:", e.stderr.decode())
        raise


def wav_to_mp3(input_path: str, skip_if_exists: bool = False) -> str:
    """
    Convert a WAV audio file to MP3 format using ffmpeg-python.
    
    :param input_path: Path to the input WAV file.
    """
    output_path = input_path.replace(".wav", ".mp3")

    if skip_if_exists and os.path.exists(output_path):
        print(f"Output MP3 file already exists at {output_path}. Skipping conversion.")
        return output_path
    
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar='44100', ac=2)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio converted successfully to {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print("Error converting audio:", e.stderr.decode())

# Example usage
# if __name__ == "__main__":
#     input_video = "S09E03 - US Deep South Road Trip.mkv"
#     convert_to_wav(input_video)


# if __name__ == '__main__':
#     wav_to_mp3("extracted_segments/SPEAKER_18_segments.wav")