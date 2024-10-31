import whisper
import torch
import os

def transcribe_audio(audio_path, model_size="medium", language=None, task="transcribe"):
    """
    Transcribe the given audio file using OpenAI's Whisper model.

    Parameters:
    - audio_path (str): Path to the audio file.
    - model_size (str): Size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large", "turbo").
    - language (str, optional): Specify the language of the audio to improve accuracy.
    - task (str): "transcribe" for transcription or "translate" to translate to English.

    Returns:
    - tuple: (transcript (str), output_path (str))
    """
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Whisper model
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size, device=device)

    # Transcribe the audio
    print(f"Transcribing '{audio_path}'...")
    result = model.transcribe(audio_path, language=language, task=task)

    # Prepare output directory and file path
    output_dir = "transcriptions"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_txt = os.path.join(output_dir, f"{base_name}.txt")

    # Save the transcription to a text file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcription complete. Results saved to '{output_txt}'.")

    return result["text"], output_txt

if __name__ == "__main__":
    # Example audio file path
    # audio_file = "isolated_vocals.wav"
    audio_file = "extracted_segments/SPEAKER_05_segments.wav"

    """
    Available model sizes:
    tiny
    base
    small
    medium
    large
    turbo
    """
    transcript, transcript_path = transcribe_audio(
        audio_file, 
        model_size="turbo", 
        language=None, 
        task="transcribe"
    )

    # Optionally, you can print the transcript and its path
    print("\nGenerated Transcript:")
    print(transcript)
    print(f"\nTranscript saved at: {transcript_path}")
