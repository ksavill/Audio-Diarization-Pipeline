import os
import sys
import torch
from pyannote.audio import Pipeline
import logging
import re

"""
TODO:
- Create an access token on Hugging Face (account required):
  https://huggingface.co/settings/tokens

- Ensure you have accepted the EULA for these models:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
"""

CITATIONS = """
CITATIONS:

@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
"""
print(CITATIONS)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("diarize.log")  # Optional: Log to a file
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """
    Replaces spaces with underscores and removes any characters that are not alphanumeric, underscores, or dots.

    Args:
        filename (str): The original filename.

    Returns:
        str: The sanitized filename.
    """
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove any other unwanted characters (optional)
    filename = re.sub(r'[^\w\.-]', '', filename)
    return filename

def get_auth_token():
    """
    Retrieves the Hugging Face authentication token from environment variables.

    Returns:
        str: Hugging Face authentication token.

    Raises:
        EnvironmentError: If the token is not found.
    """
    auth_token = os.getenv("HF_AUTH_TOKEN")
    if not auth_token:
        logger.error("HF_AUTH_TOKEN environment variable is not set.")
        raise EnvironmentError("HF_AUTH_TOKEN environment variable is not set.")
    return auth_token

def initialize_pipeline(auth_token, device):
    """
    Initializes and returns the pyannote speaker diarization pipeline.

    Args:
        auth_token (str): Hugging Face authentication token.
        device (torch.device): Device to run the pipeline on.

    Returns:
        Pipeline: Initialized pyannote pipeline.
    """
    try:
        # Instantiate the pipeline with authentication
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        logger.info(f"Pipeline loaded successfully on {device.type.upper()}.")
    except Exception as e:
        logger.error(f"Error loading the pipeline: {e}")
        raise

    try:
        # Move the pipeline to the specified device (CPU or CUDA)
        pipeline.to(device)
        logger.info(f"Pipeline moved to {device.type.upper()} successfully.")
    except Exception as e:
        logger.error(f"Error moving the pipeline to {device.type.upper()}: {e}")
        raise

    return pipeline

def perform_diarization(
        pipeline: Pipeline,
        audio_path: str,
        skip_if_exists: bool = False,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> str:
    """
    Performs speaker diarization on a given audio file and saves the result to an RTTM file.

    Parameters:
    - pipeline (Pipeline): Initialized pyannote pipeline.
    - audio_path (str): Path to the audio file.
    - skip_if_exists (bool): If True, skip diarization if the RTTM file already exists.
    - num_speakers (int, optional): Exact number of speakers. If set, min_speakers and max_speakers are ignored.
    - min_speakers (int, optional): Minimum number of speakers.
    - max_speakers (int, optional): Maximum number of speakers.

    Returns:
    - str: Path to the saved RTTM file.
    """
    
    output_folder = "diarized_audio"
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    sanitized_name = sanitize_filename(base_name)
    output_path = os.path.join(output_folder, f"{sanitized_name}.rttm")
    
    # Check if RTTM file exists and skip if required
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"RTTM file already exists at '{output_path}'. Skipping diarization.")
        return output_path

    # Prepare the parameters for diarization
    diarization_params = {}
    if num_speakers is not None:
        diarization_params["num_speakers"] = num_speakers
        logger.info(f"Setting number of speakers to {num_speakers}. Ignoring min_speakers and max_speakers.")
    else:
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
            logger.info(f"Setting minimum number of speakers to {min_speakers}.")
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
            logger.info(f"Setting maximum number of speakers to {max_speakers}.")

    # Create input dictionary with sanitized URI
    input_data = {
        "uri": sanitized_name,
        "audio": audio_path
    }

    try:
        # Perform diarization with the specified parameters
        logger.info(f"Starting diarization on '{audio_path}' with parameters: {diarization_params}...")
        diarization = pipeline(input_data, **diarization_params)
        logger.info("Diarization completed successfully.")
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        raise

    # Save diarization results in RTTM format
    try:
        with open(output_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        logger.info(f"RTTM file saved successfully as '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to write RTTM file: {e}")
        raise

    return output_path

def main():
    """
    Example usage of the diarize module.
    """
    
    audio_file = "wav_conversions/S15E04.wav"  # Update with actual audio file path
    
    # Retrieve Hugging Face auth token
    try:
        auth_token = get_auth_token()
    except EnvironmentError as e:
        logger.error(e)
        sys.exit(1)
    
    # Determine the device to use (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("CUDA is available. Using GPU for processing.")
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
    
    # Initialize the pipeline
    try:
        pipeline = initialize_pipeline(auth_token, device)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Perform diarization
    try:
        diarization = perform_diarization(pipeline, audio_file)
    except Exception as e:
        logger.error(f"Failed to perform diarization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()