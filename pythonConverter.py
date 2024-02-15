import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def fir_comb_filter(input_signal, sample_rate=44100, gain=0.5, delay_sec=0.001):
    delay_samples = int(sample_rate * delay_sec)
    output_signal = np.zeros_like(input_signal)
    for channel in range(input_signal.shape[1]):  # Iterate over channels
        for n in range(len(input_signal)):
            output_signal[n, channel] = input_signal[n, channel]
            if n - delay_samples >= 0:
                output_signal[n, channel] += (
                    gain * input_signal[n - delay_samples, channel]
                )
    return output_signal


def iir_comb_filter(input_signal, sample_rate, gain, delay_sec):
    delay_samples = int(sample_rate * delay_sec)
    output_signal = np.zeros_like(input_signal)
    for channel in range(input_signal.shape[1]):  # Iterate over channels
        for n in range(len(input_signal)):
            output_signal[n, channel] = input_signal[n, channel]
            if n - delay_samples >= 0:
                output_signal[n, channel] += (
                    gain * output_signal[n - delay_samples, channel]
                )
    return output_signal


def process_audio_file(
    file_path,
    output_path,
    sample_rate_new=44100,
    gain=0.5,
    delay_sec=0.25,
    filter_type="fir",
):
    # Read the input file
    input_signal, original_sample_rate = librosa.load(file_path, sr=sample_rate_new, mono=False)
    print(input_signal.ndim)
    # Check if input_signal is mono and convert to 2D array if it is
    if input_signal.ndim == 1:
        input_signal = input_signal[np.newaxis, :]
    # Resample if needed
    if original_sample_rate != sample_rate_new:
        input_signal = librosa.resample(input_signal, orig_sr=original_sample_rate, target_sr=sample_rate_new)

    # Apply comb filter
    if filter_type == "fir":
        output_signal = fir_comb_filter(input_signal, sample_rate_new, gain, delay_sec)
    else:  # Assume IIR if not FIR
        output_signal = iir_comb_filter(input_signal, sample_rate_new, gain, delay_sec)

    # Write the output file
    # Assuming output is stereo; adjust accordingly if not
    if output_signal.shape[1] == 1:  # Convert back to mono for writing if necessary
        output_signal = output_signal.ravel()
    # output_signal = np.clip(
    #     output_signal, -1.0, 1.0
    # )  # Ensure signal is within float32 range
    output_signal_int16 = (output_signal * 32767).astype(
        np.int16
    )  # Convert to int16 for WAV file
    if output_signal_int16.ndim > 1:
        output_signal_int16 = output_signal_int16.T
    sf.write(output_path, output_signal_int16, sample_rate_new)


def compare_audio_files(file1_path, file2_path):
    # Load the audio files
    
    y1, sr1 = librosa.load(file1_path, sr=None, mono=False)
    y2, sr2 = librosa.load(file2_path, sr=None, mono=False)
    print(y1.shape, y2.shape, sr1, sr2)
    # Ensure both files have the same sample rate and number of channels
    assert sr1 == sr2, "Sample rates differ between the two audio files."
    assert y1.shape == y2.shape, "Audio dimensions differ between the two audio files."

    # Compute the difference
    difference = y1 - y2
    print(difference)
    return difference, sr1


def plot_difference(
    difference,
    sample_rate,
    title="Difference between audio files",
    save_path="difference_plot.png",
):
    # Assuming the difference array could be multi-channel, we plot each channel separately
    if difference.ndim == 1:
        difference = difference[np.newaxis, :]
    channels = difference.shape[0]
    time = np.arange(difference.shape[1]) / sample_rate

    fig, axs = plt.subplots(channels, 1, figsize=(10, 4 * channels), sharex=True)
    fig.suptitle(title)

    if channels == 1:
        axs.plot(time, difference[0, :])
        axs.set_title("Channel 1")
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Amplitude")
    else:
        for i in range(channels):
            axs[i].plot(time, difference[i, :])
            axs[i].set_title(f"Channel {i + 1}")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)  # Save the plot to a file
    plt.show()


input_file = "hardrocklogo.wav"
output_file_fir = "hardrocklogo_fir_python_default.wav"
output_file_iir = "hardrocklogo_iir_python_default.wav"
fir_comp = "hardrocklogo_fir_rust_default.wav"
iir_comp = "hardrocklogo_iir_rust_default.wav"

process_audio_file(input_file, output_file_fir, 48000, 0.5, 0.001, 'fir')
process_audio_file(input_file, output_file_iir, 48000, 0.5, 0.001, 'iir')
difference_fir, sr = compare_audio_files(fir_comp, output_file_fir)
difference_iir, sr = compare_audio_files(iir_comp, output_file_iir)
plot_difference(difference_fir, sr, title=fir_comp, save_path="hardrock_fir_diff.png")
plot_difference(difference_iir, sr, title=iir_comp, save_path="hardrock_iir_diff.png")
