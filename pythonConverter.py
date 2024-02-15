import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Step 1: Load WAV files
def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

# Step 2: Compute the difference
def compute_difference(data1, data2):
    return data1 - data2

# Step 3: Save the difference to a text file
def save_difference_to_file(difference, file_name="difference.txt"):
    np.savetxt(file_name, difference, fmt='%d')

# Step 4: Generate and save plot
def plot_and_save_difference(difference, sample_rate, file_name="difference_plot_IIR.png"):
    time = np.arange(difference.size) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, difference)
    plt.title("Difference Between Two IIR WAV Files")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude Difference")
    plt.savefig(file_name, dpi=300)
    plt.close()

# Main execution
if __name__ == "__main__":
    file1 = "hardrocklogo_fir_correct_default.wav"
    file2 = "hardrocklogo_fir_rust_default.wav"

    sample_rate1, data1 = load_wav(file1)
    sample_rate2, data2 = load_wav(file2)

    # Ensure the files have the same sample rate
    if sample_rate1 != sample_rate2:
        raise ValueError("The WAV files have different sample rates.")

    # Handle stereo files by using only one channel
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if len(data2.shape) > 1:
        data2 = data2[:, 0]

    difference = compute_difference(data1, data2)
    save_difference_to_file(difference)
    plot_and_save_difference(difference, sample_rate1)
