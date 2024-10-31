import sys
from pydub import AudioSegment
import pandas as pd

def load_diarization(rttm_path):
    """
    Parses an RTTM file and returns a DataFrame with diarization segments.
    
    Parameters:
        rttm_path (str): Path to the RTTM file.
    
    Returns:
        pd.DataFrame: DataFrame containing FileID, Channel, Speaker, Onset, Offset.
    """
    data = {
        'FileID': [],
        'Channel': [],
        'Speaker': [],
        'Onset': [],
        'Offset': []
    }
    
    with open(rttm_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            parts = line.split()
            
            if len(parts) < 9:
                print(f"Warning: Line {line_num} is malformed: '{line}'")
                continue  # Skip malformed lines
            
            entry_type = parts[0]
            if entry_type != 'SPEAKER':
                continue  # Skip non-speaker entries
            
            try:
                file_id = parts[1]
                channel = int(parts[2])
                onset = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                offset = onset + duration
            except ValueError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue  # Skip lines with parsing errors
            
            # Append to data dictionary
            data['FileID'].append(file_id)
            data['Channel'].append(channel)
            data['Speaker'].append(speaker)
            data['Onset'].append(onset)
            data['Offset'].append(offset)
    
    df = pd.DataFrame(data)
    return df

def get_all_speakers(diarization_df):
    """
    Retrieves a list of all unique speakers from the diarization DataFrame.
    
    Parameters:
        diarization_df (pd.DataFrame): DataFrame containing diarization segments.
    
    Returns:
        list: List of unique speaker labels.
    """
    speakers = diarization_df['Speaker'].unique().tolist()
    print(f"Found {len(speakers)} unique speaker(s): {', '.join(speakers)}")
    return speakers

def extract_single_segment(diarization_df, audio_path, output_path, speaker=None, index=None):
    """
    Extracts a single audio segment based on speaker label or segment index.

    Parameters:
    - diarization_df (pd.DataFrame): DataFrame with diarization segments.
    - audio_path (str): Path to the original audio file.
    - output_path (str): Path to save the extracted audio segment.
    - speaker (str, optional): Speaker label to extract the first segment from.
    - index (int, optional): Segment index to extract (0-based).

    Raises:
    - ValueError: If neither speaker nor index is specified.
    - SystemExit: If extraction fails.
    """
    if speaker is None and index is None:
        raise ValueError("Please specify either a speaker label or a segment index.")

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

    selected_row = None
    speaker_label = None

    if index is not None:
        if index < 0 or index >= len(diarization_df):
            print(f"Index {index} is out of range. Total segments: {len(diarization_df)}.")
            sys.exit(1)
        selected_row = diarization_df.iloc[index]
        speaker_label = selected_row['Speaker']
        print(f"Extracting segment {index}: Speaker {speaker_label}, "
              f"Start {selected_row['Onset']:.2f}s, End {selected_row['Offset']:.2f}s.")
    elif speaker is not None:
        # Find the first segment for the specified speaker
        speaker_rows = diarization_df[diarization_df['Speaker'] == speaker]
        if speaker_rows.empty:
            print(f"No segments found for speaker '{speaker}'.")
            sys.exit(1)
        selected_row = speaker_rows.iloc[0]
        speaker_label = selected_row['Speaker']
        print(f"Extracting first segment of Speaker {speaker_label}: "
              f"Start {selected_row['Onset']:.2f}s, End {selected_row['Offset']:.2f}s.")

    # Extract the audio segment
    start_ms = int(selected_row['Onset'] * 1000)
    end_ms = int(selected_row['Offset'] * 1000)
    extracted_audio = audio[start_ms:end_ms]

    # Export the extracted segment
    try:
        extracted_audio.export(output_path, format="wav")
        print(f"Extracted audio segment saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving extracted audio: {e}")
        sys.exit(1)

def extract_all_speaker_segments(diarization_df, audio_path, output_dir, speaker):
    """
    Extracts all audio segments of a specific speaker and combines them into one audio file.
    The output filename will include the speaker's label.

    Parameters:
    - diarization_df (pd.DataFrame): DataFrame with diarization segments.
    - audio_path (str): Path to the original audio file.
    - output_dir (str): Directory to save the combined audio segments.
    - speaker (str): Speaker label to extract.

    Raises:
    - SystemExit: If no segments are found for the specified speaker or extraction fails.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

    combined = AudioSegment.empty()
    speaker_rows = diarization_df[diarization_df['Speaker'] == speaker]

    if speaker_rows.empty:
        print(f"No segments found for speaker '{speaker}'.")
        sys.exit(1)

    for idx, row in speaker_rows.iterrows():
        start_ms = int(row['Onset'] * 1000)
        end_ms = int(row['Offset'] * 1000)
        segment_audio = audio[start_ms:end_ms]
        combined += segment_audio
        print(f"Added segment: Speaker {speaker}, Start {row['Onset']:.2f}s, End {row['Offset']:.2f}s.")

    # Ensure the output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output filename with the speaker label
    output_filename = f"{speaker}_segments.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Export the combined audio
    try:
        combined.export(output_path, format="wav")
        print(f"All segments for speaker '{speaker}' saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving combined audio: {e}")
        sys.exit(1)

# Usage Example
if __name__ == "__main__":
    rttm_path = 'output.rttm'           # Replace with your RTTM file path
    audio_path = 'processed_audio.wav'  # Replace with your audio file path
    output_dir = 'extracted_segments'   # Directory to save extracted segments

    # Load diarization data
    diarization_df = load_diarization(rttm_path)
    print(diarization_df)

    # Get all unique speakers
    speakers = get_all_speakers(diarization_df)

    # Extract a single segment by index
    # extract_single_segment(diarization_df, audio_path, 'output_segment.wav', index=0)
    
    # Alternatively, extract a single segment by speaker
    # extract_single_segment(diarization_df, audio_path, 'speaker_first_segment.wav', speaker='SPEAKER_18')

    extract_all_speaker_segments(diarization_df, audio_path, output_dir, 'SPEAKER_18')
    extract_all_speaker_segments(diarization_df, audio_path, output_dir, 'SPEAKER_05')
    # Extract all segments for each speaker and save with speaker label in filename
    # for speaker in speakers:
    #     extract_all_speaker_segments(diarization_df, audio_path, output_dir, speaker)
