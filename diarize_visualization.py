import sys
import torch
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pandas as pd
from process_diarization import load_diarization
import matplotlib.patches as mpatches

def visualize_diarization(diarization_df, show_plot=True):
    """
    Visualizes speaker diarization from a DataFrame.

    Parameters:
    - diarization_df (pd.DataFrame): DataFrame containing diarization segments.
    - show_plot (bool, optional): If True, displays the plot. Defaults to True.

    Raises:
    - Exception: For errors during visualization.
    """
    try:
        df = diarization_df.copy()
        
        # Sort segments by onset time
        df = df.sort_values(by='Onset')
        
        # Get unique speakers
        speakers = df['Speaker'].unique()
        num_speakers = len(speakers)

        # Assign a unique color to each speaker
        colors = plt.cm.get_cmap('tab20', num_speakers)
        speaker_colors = {speaker: colors(i) for i, speaker in enumerate(speakers)}

        fig, ax = plt.subplots(figsize=(12, 2 + num_speakers))

        # Y-axis positions for each speaker
        y_positions = {speaker: i for i, speaker in enumerate(speakers)}

        for _, row in df.iterrows():
            speaker = row['Speaker']
            onset = row['Onset']
            duration = row['Offset'] - row['Onset']
            ax.broken_barh(
                [(onset, duration)],
                (y_positions[speaker] - 0.4, 0.8),
                facecolors=speaker_colors[speaker]
            )

        # Configure Y-axis
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel('Time (s)')
        ax.set_title('Speaker Diarization')

        # Create legend
        patches = [
            mpatches.Patch(color=color, label=speaker)
            for speaker, color in speaker_colors.items()
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

def get_device():
    """
    Determines the computation device (CUDA if available, else CPU).

    Returns:
    - torch.device: The device to use for computation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available. Using GPU for processing.")
    else:
        print("CUDA is not available. Using CPU for processing.")
    return device

def main_visualize_diarization(rttm_path, show_plot=True):
    """
    Convenience function to visualize diarization.

    Parameters:
    - rttm_path (str): Path to the RTTM file containing diarization results.
    - show_plot (bool, optional): If True, displays the plot. Defaults to True.
    """
    # Load diarization from RTTM
    try:
        diarization_df = load_diarization(rttm_path)
    except Exception as e:
        print(f"Error loading diarization: {e}")
        sys.exit(1)

    # Plot the diarization
    visualize_diarization(diarization_df, show_plot)

def main():
    rttm_file = "diarized_audio/S15E04.rttm"  # Path to your RTTM file

    # Visualize the diarization
    main_visualize_diarization(rttm_path=rttm_file, show_plot=True)

if __name__ == "__main__":
    main()
