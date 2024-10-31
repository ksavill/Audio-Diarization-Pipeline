import os
import sys
import logging
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import input_to_wav
import background_noise_reduction
import diarize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize a semaphore for GPU-bound tasks
# Set to 1 or 2 based on GPU's capacity (8GB VRAM)
GPU_SEMAPHORE = threading.Semaphore(2)  # Allow up to 2 concurrent GPU tasks

def process_single_file(filename, input_dir, pipeline):
    file_path = os.path.join(input_dir, filename)

    if not os.path.isfile(file_path):
        logger.warning(f"Skipping non-file: {file_path}")
        return

    try:
        # Step 1: Convert to WAV format (CPU-Bound)
        wav_path = input_to_wav.convert_to_wav(file_path, skip_if_exists=True)
        logger.info(f"Converted to WAV: {wav_path}")

        # Step 2: Apply background noise reduction (GPU-Bound)
        # Acquire semaphore before GPU task
        with GPU_SEMAPHORE:
            logger.info(f"Acquired GPU semaphore for noise reduction: {wav_path}")
            noise_reduced_path = background_noise_reduction.remove_background_audio(wav_path, skip_if_exists=True)
            logger.info(f"Background noise reduced: {noise_reduced_path}")

        # Step 3: Perform diarization (GPU-Bound)
        # Acquire semaphore before GPU task
        with GPU_SEMAPHORE:
            logger.info(f"Acquired GPU semaphore for diarization: {noise_reduced_path}")
            diarization_path = diarize.perform_diarization(pipeline, noise_reduced_path, skip_if_exists=True)
            logger.info(f"Diarization completed and saved: {diarization_path}")

    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")

def process_audio_pipeline(input_dir, max_workers=4):
    # Initialize diarization pipeline
    try:
        auth_token = diarize.get_auth_token()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            logger.info("CUDA is available. GPU tasks will be accelerated.")
        else:
            logger.warning("CUDA is not available. GPU-bound tasks may run slower or fail.")

        pipeline = diarize.initialize_pipeline(auth_token, device)
    except Exception as e:
        logger.error(f"Failed to initialize diarization pipeline: {e}")
        sys.exit(1)

    # List all files in the input directory
    filenames = os.listdir(input_dir)

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_file, 
                filename, 
                input_dir, 
                pipeline
            )
            for filename in filenames
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Exception occurred during processing: {e}")

if __name__ == "__main__":
    input_directory = "Top Gear Specials"
    process_audio_pipeline(input_directory)