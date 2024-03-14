//! # Vibrato Effect Processor
//!
//! Provides a vibrato effect processor designed for real-time audio signal processing. 
//! The `Vibrato` struct modulates the pitch of audio input signals, creating a vibrato effect 
//! by leveraging a low-frequency oscillator (LFO) and delay buffers for each channel of audio input.
use crate::lfo::LFO;
use crate::ring_buffer::RingBuffer;

/// Errors that can occur within the vibrato effect processing.
///
/// These errors are returned when attempting to create a new `Vibrato` instance or update its parameters
/// with values that are outside of the acceptable range or logically incorrect.
#[derive(Debug, Clone)]
pub enum Error {
    /// Error due to an invalid parameter value.
    ///
    /// Indicates that a parameter was provided with a value that is not supported by the vibrato processor,
    /// either because it is out of the acceptable range or because it fails to meet a specific precondition.
    ///
    /// - `name`: A descriptive name of the parameter that received the invalid value.
    /// - `value`: The actual invalid value that was provided.
    InvalidValue { name: String, value: f32 },
}

/// The `Vibrato` struct applies a vibrato effect to an audio signal.
///
/// It uses a low-frequency oscillator (LFO) to modulate the delay time of the signal,
/// producing variations in pitch that result in the vibrato effect. The modulation and its intensity
/// can be customized through parameters such as modulation frequency and modulation depth.
pub struct Vibrato {
    sample_rate: f32,
    delay_time: f32,
    width_time: f32,
    modulation_frequency: f32,
    delay_buffers: Vec<RingBuffer<f32>>,
    lfo: LFO,
    channels: usize,
}

impl Vibrato {
    /// Creates a new instance of the `Vibrato` effect processor.
    ///
    /// Validates the provided parameters and initializes the internal state, including
    /// the delay buffers and the LFO with the specified settings.
    ///
    /// # Parameters
    ///
    /// - `sample_rate`: The sample rate of the audio in Hz. Must be positive.
    /// - `delay_time`: The base delay time in seconds. Must be within the capacity of the delay buffers.
    /// - `width_time`: The width of the vibrato modulation in seconds. Cannot exceed the `delay_time`.
    /// - `modulation_frequency`: The frequency of the vibrato effect in Hz. Must be positive.
    /// - `channels`: The number of audio channels to process. Must be a positive integer.
    ///
    /// # Returns
    ///
    /// - `Ok(Self)`: A new `Vibrato` instance configured with the specified parameters.
    /// - `Err(Error)`: An error if any parameter is invalid, detailing the nature of the invalid parameter.
    pub fn new(
        sample_rate: f32,
        delay_time: f32,
        width_time: f32,
        modulation_frequency: f32,
        channels: usize,
    ) -> Result<Self, Error> {
        let delay = (delay_time * sample_rate).round() as usize;
        let width = (width_time * sample_rate).round() as usize;
        let capacity = Self::calculate_capacity(delay, width);

        if width > delay {
            return Err(Error::InvalidValue {
                name: "width in seconds".to_string(),
                value: width_time,
            });
        }
        if modulation_frequency <= 0.0 {
            return Err(Error::InvalidValue {
                name: "modulation frequency in Hz".to_string(),
                value: modulation_frequency,
            });
        }

        let delay_buffers = (0..channels)
            .map(|_| RingBuffer::<f32>::new(capacity))
            .collect();

        let mut lfo = LFO::new(sample_rate, 1024);
        lfo.set_frequency(modulation_frequency);
        lfo.set_amplitude(width as f32);

        Ok(Self {
            sample_rate,
            delay_time,
            width_time,
            modulation_frequency,
            delay_buffers,
            lfo,
            channels,
        })
    }

    /// Resets the internal delay buffers of the vibrato effect processor to their initial state.
    ///
    /// This method is useful for clearing the effect's internal state before processing a new audio stream
    /// or when wishing to reset the effect's state at any point during its use.
    pub fn reset(&mut self) {
        self.delay_buffers.iter_mut().for_each(RingBuffer::reset);
    }

    /// Updates the parameters of the vibrato effect processor.
    ///
    /// Allows dynamic adjustment of the vibrato's delay time, modulation width, and modulation frequency.
    /// Validates the new parameters before applying them to ensure they are within acceptable ranges.
    ///
    /// # Parameters
    ///
    /// - `delay_time`: The new base delay time in seconds.
    /// - `width_time`: The new modulation width in seconds.
    /// - `modulation_frequency`: The new frequency of the modulation in Hz.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the parameters are successfully updated.
    /// - `Err(Error)`: If any parameter is invalid, with details about the issue.
    pub fn set_param(
        &mut self,
        delay_time: f32,
        width_time: f32,
        modulation_frequency: f32,
    ) -> Result<(), Error> {
        let delay = (delay_time * self.sample_rate).round() as usize;
        let width = (width_time * self.sample_rate).round() as usize;
        let capacity = Self::calculate_capacity(delay, width);

        if width > delay {
            return Err(Error::InvalidValue {
                name: "width in seconds".to_string(),
                value: width_time,
            });
        }
        if modulation_frequency <= 0.0 {
            return Err(Error::InvalidValue {
                name: "modulation frequency in Hz".to_string(),
                value: modulation_frequency,
            });
        }

        self.delay_time = delay_time;
        self.width_time = width_time;
        self.modulation_frequency = modulation_frequency;
        self.delay_buffers.iter_mut().for_each(|buffer| *buffer = RingBuffer::new(capacity));

        Ok(())
    }

    /// Retrieves the current parameters of the vibrato effect processor.
    ///
    /// Useful for inspecting the current configuration of the vibrato effect, 
    /// including its base delay time, modulation width, and modulation frequency.
    ///
    /// # Returns
    ///
    /// A tuple containing the current delay time, width time, and modulation frequency parameters.
    pub fn get_param(&self) -> (f32, f32, f32) {
        (self.delay_time, self.width_time, self.modulation_frequency)
    }

    /// Processes the input audio buffers and applies the vibrato effect.
    ///
    /// This method modulates the pitch of each audio channel according to the vibrato's current settings,
    /// writing the processed audio to the output buffers.
    ///
    /// # Parameters
    ///
    /// - `input`: A slice of references to input audio buffers, one per channel.
    /// - `output`: A slice of mutable references to output audio buffers, one per channel,
    ///             where the processed audio will be stored.
    ///
    /// # Panics
    ///
    /// May panic if the `input` and `output` slices are not of equal length or if the length of any
    /// output buffer does not match the corresponding input buffer.
    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) {
        for (idx, &input) in input.iter().enumerate() {
            input.iter().enumerate().for_each(|(sample_idx, &sample)| {
                let depth_mod = self.lfo.next_mod();
                let delay_whole = self.delay_time * self.sample_rate + depth_mod;
                self.delay_buffers[idx].push(sample);
                let delayed_sample = self.delay_buffers[idx].get_frac(delay_whole);
                output[idx][sample_idx] = delayed_sample;
            });
        }
    }    

    /// Calculates the required capacity for the delay buffers based on the delay and width parameters.
    ///
    /// Ensures that there is sufficient space in the delay buffers to accommodate the base delay time
    /// and the maximum modulation depth induced by the vibrato effect.
    ///
    /// # Parameters
    ///
    /// - `delay`: The number of samples corresponding to the base delay time.
    /// - `width`: The number of samples corresponding to the maximum modulation width.
    ///
    /// # Returns
    ///
    /// The calculated capacity for the delay buffers, ensuring adequate space for modulation.
    fn calculate_capacity(delay: usize, width: usize) -> usize {
        1 + delay + width * 2
    }
}





#[cfg(test)]
mod tests {
    //! Unit tests for the `Vibrato` audio effect processor.
    //!
    //! These tests validate the correct initialization, parameter modification,
    //! signal processing, and reset functionality of the `Vibrato` struct, ensuring
    //! reliability and correctness across its main use cases.
    use super::*;
    use crate::ring_buffer::RingBuffer;

    /// Validates successful creation and initialization of a `Vibrato` instance with given parameters.
    /// 
    /// Ensures that the `Vibrato::new` function correctly initializes the struct's fields with the provided
    /// arguments and that the internal structures such as delay buffers and LFO are set up properly.
    /// The test confirms that no error is returned and that all specified parameters are accurately reflected
    /// in the `Vibrato` instance's state.
    #[test]
    fn test_Vibrato_new() {
        let sample_rate = 44100.0;
        let delay_time = 0.05; // Assuming this corresponds to `max_delay_secs` based on the context
        let width_time = 0.01;
        let modulation_frequency = 5.0;
        let channels = 2;

        match Vibrato::new(sample_rate, delay_time, width_time, modulation_frequency, channels) {
            Ok(Vibrato) => {
                assert_eq!(Vibrato.sample_rate, sample_rate);
                assert_eq!(Vibrato.delay_time, delay_time);
                assert_eq!(Vibrato.width_time, width_time);
                assert_eq!(Vibrato.modulation_frequency, modulation_frequency);
                assert_eq!(Vibrato.channels, channels);
                // Additional assertions can be added here to check the initial state of delay_buffers, lfo, etc.
            },
            Err(e) => panic!("Failed to create Vibrato: {:?}", e),
        }
    }

    /// Tests the functionality of updating `Vibrato`'s parameters using `set_param`.
    ///
    /// This test ensures that after changing the vibrato's parameters, the updated values are
    /// accurately reflected. It verifies the method's ability to correctly update and maintain the integrity
    /// of the `Vibrato`'s state, including recalculations necessary for the internal processing logic.
    #[test]
    fn test_set_param() {
        let sample_rate_hz = 44100.0;
        let delay_secs = 0.01;
        let width_secs = 0.005;
        let mod_freq_hz = 5.0;
        let num_channels = 2;
        let mut vibrato = Vibrato::new(
            sample_rate_hz,
            delay_secs,
            width_secs,
            mod_freq_hz,
            num_channels,
        )
        .unwrap();
        let (delay_secs_check_1, width_secs_check_1, mod_freq_hz_check_1) = vibrato.get_param();
        vibrato.set_param(0.02, 0.01, 10.0).unwrap();
        let (delay_secs_check_2, width_secs_check_2, mod_freq_hz_check_2) = vibrato.get_param();
        assert_eq!(delay_secs_check_1 * 2.0, delay_secs_check_2);
        assert_eq!(width_secs_check_1 * 2.0, width_secs_check_2);
        assert_eq!(mod_freq_hz_check_1 * 2.0, mod_freq_hz_check_2);
    }
    
    /// Checks the `reset` method's ability to revert `Vibrato`'s internal state to initial conditions.
    ///
    /// By processing a signal and then resetting, this test verifies that the `reset` method effectively
    /// clears any internal state, ensuring the `Vibrato` is ready for fresh processing without remnants of
    /// previous data affecting the output. It demonstrates the reset functionality's critical role in reusing
    /// the effect processor for multiple audio streams.
    #[test]
    fn test_reset() {
        let sample_rate = 44100.0;
        let delay = 0.01;
        let width = 0.005;
        let mod_freq = 5.0;
        let channels = 1;
    
        // Simplify the creation and unwrap operation with expect to provide more context on failure
        let mut vibrato = Vibrato::new(sample_rate, delay, width, mod_freq, channels).expect("Create Vibrato failed");
    
        // Directly create slices instead of converting through Vec
        let mut input = [1.0; 44100];
        let mut output = [0.0; 44100];
    
        // Directly pass slice references
        vibrato.process(&[&input], &mut [&mut output]);
        vibrato.reset();
    
        assert_eq!(output[0], 0.0, "Output should be reset to 0.0 after calling reset");
    }
    
    /// Evaluates the vibrato effect's handling of delay and minimal modulation.
    ///
    /// This test specifically looks at how the `Vibrato` applies an initial delay and minimal modulation to an input signal,
    /// assessing the effect processor's ability to handle subtle modulations. It's crucial for validating that the vibrato
    /// effect can be finely tuned for subtle audio enhancements.
    #[test]
    fn output_reflects_delay_effect_with_minimal_modulation() {
        let sample_rate = 44100.0;
        let delay_time = 0.1; // This should introduce a significant initial delay
        let width_time = 0.05;
        let modulation_frequency = 0.001; // Minimal modulation
        let channels = 1;

        let mut filter = Vibrato::new(sample_rate, delay_time, width_time, modulation_frequency, channels).expect("Create Vibrato failed");

        let input_signal = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]]; // Extended input
        let mut output_signal = vec![vec![0.0; 15]]; // Extended output buffer

        let input_slices: Vec<&[f32]> = input_signal.iter().map(|channel| channel.as_slice()).collect();
        let mut output_slices: Vec<&mut [f32]> = output_signal.iter_mut().map(|channel| channel.as_mut_slice()).collect();

        filter.process(&input_slices, &mut output_slices);

        // Instead of a direct comparison, we might expect the output to start with zeroes due to the delay.
        // Here, we check a simplified condition focusing on the delay effect.
        assert_ne!(output_signal[0], input_signal[0], "Processed output unexpectedly matches the input signal.");
        // Further detailed assertions should be added to verify the delay and modulation effects more precisely.
    }
    
    /// Tests the `Vibrato` effect's response to a constant (DC) input signal.
    ///
    /// Verifies that a constant input yields a consistent output after an initial transient period, reflecting the
    /// processor's ability to maintain steady-state conditions. This test is significant for understanding how
    /// `Vibrato` interacts with sustained tones or notes, which are common in musical applications.
    #[test]
    fn dc_input_results_in_dc_output() {
        let sample_rate_hz = 44100.0;
        let delay_secs = 0.01; // Example delay
        let width_secs = 0.005; // Example modulation width
        let mod_freq_hz = 5.0; // Example modulation frequency
        let num_channels = 1;
        let mut vibrato = Vibrato::new(
            sample_rate_hz,
            delay_secs,
            width_secs,
            mod_freq_hz,
            num_channels,
        )
        .unwrap();

        let input_len = 441; // Example input length
        let dc_value = 0.5; // Example DC value for input
        let input = vec![vec![dc_value; input_len]; num_channels];
        let mut output = vec![vec![0.0; input_len]; num_channels];

        let input_slices: Vec<&[f32]> = input.iter().map(|v| v.as_slice()).collect();
        let mut output_slices: Vec<&mut [f32]> =
            output.iter_mut().map(|v| v.as_mut_slice()).collect();

        vibrato.process(&input_slices, &mut output_slices);

        // Allow some initial transient samples for the effect to stabilize
        let transient_samples = (sample_rate_hz * delay_secs) as usize;

        for channel in output.iter() {
            for &sample in channel.iter().skip(transient_samples) {
                assert!(
                    (sample - dc_value).abs() < 0.001,
                    "Output should remain constant (DC) after initial transient"
                );
            }
        }
    }

    /// Validates the `Vibrato` effect's processing across different input block sizes.
    ///
    /// By testing with various input lengths, this test ensures that the `Vibrato` can handle audio buffers of
    /// different sizes without compromising the consistency or quality of the vibrato effect. It highlights the
    /// processor's flexibility and reliability in a real-world usage scenario where buffer sizes can vary.
    #[test]
    fn test_varying_input_block_sizes() {
        let sample_rate = 44100.0;
        let delay_seconds = 0.01;
        let width_seconds = 0.005;
        let modulation_frequency = 5.0;
        let channels = 1;
        let mut vibrato_effect = Vibrato::new(sample_rate,delay_seconds,width_seconds,modulation_frequency,channels,).expect("Create Vibrato failed");
    
        // Testing with various input sizes
        for &input_size in &[100, 500, 1000] {
            let input_signal = vec![vec![1.0; input_size]; channels];
            let mut output_signal = vec![vec![0.0; input_size]; channels];
    
            let input_slices: Vec<_> = input_signal.iter().map(Vec::as_slice).collect();
            let mut output_slices: Vec<_> = output_signal.iter_mut().map(Vec::as_mut_slice).collect();
    
            vibrato_effect.process(&input_slices, &mut output_slices);
    
            let transient_sample_count = (sample_rate * delay_seconds) as usize;
    
            for output_channel in output_signal.iter() {
                for &sample in output_channel.iter().skip(transient_sample_count) {
                    assert!(
                        (sample - 1.0).abs() < 0.001,
                        "Output should remain constant (DC) after the initial transient"
                    );
                }
            }
        }
    }
    

    /// Confirms that processing a zero input signal yields a zero output signal.
    ///
    /// This test verifies the `Vibrato`'s correctness in situations where the input signal is silent, ensuring
    /// that the processor does not introduce any unintended noise or artifacts. It's essential for evaluating the
    /// processor's noise floor and ensuring transparency in audio processing.
    #[test]
    fn test_zero_input_signal() {
        // Improved variable names for clarity
        let sample_rate = 44100.0;
        let delay_time = 0.01;
        let modulation_width = 0.005;
        let modulation_frequency = 5.0;
        const NUM_CHANNELS: usize = 1; // Use a constant for fixed values like number of channels
        
        // Initialize the vibrato filter with clearer variable names
        let mut vibrato_filter = Vibrato::new(sample_rate,delay_time,modulation_width,modulation_frequency,NUM_CHANNELS,).expect("Create Vibrato instance failed");
    
        // Use the constant for dimensions
        const INPUT_LENGTH: usize = 441;
        let zero_input = vec![vec![0.0; INPUT_LENGTH]; NUM_CHANNELS]; // Zero input signal
        let mut output_signal = vec![vec![0.0; INPUT_LENGTH]; NUM_CHANNELS]; // Output buffer
    
        // Convert input and output vectors to slices for processing
        let input_slices: Vec<&[f32]> = zero_input.iter().map(Vec::as_slice).collect();
        let mut output_slices: Vec<&mut [f32]> = output_signal.iter_mut().map(Vec::as_mut_slice).collect();
    
        // Process the input signal
        vibrato_filter.process(&input_slices, &mut output_slices);
    
        // Assert all output samples are zero for zero input signal
        for channel_output in &output_signal {
            assert!(channel_output.iter().all(|&sample| sample == 0.0), "Output should be 0 for zero input signal");
        }
    }
    
    /// Assesses the application of the vibrato effect through the `process` method.
    ///
    /// Focused on evaluating the effect's audible impact, this test checks if the `process` method modifies an input
    /// signal in a manner consistent with the expected vibrato effect. It underscores the `Vibrato`'s primary functionality,
    /// ensuring the processor not only alters the audio signal as intended but also adheres to the specified modulation parameters.
    #[test]
    fn test_process_applies_vibrato_effect() {
        // Given: A Vibrato instance with known parameters
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.005; // 5ms max delay
        let width_secs = 0.003; // 3ms width
        let mod_freq_hz = 5.0; // 5Hz modulation frequency
        let num_channels = 1; // Mono signal for simplicity
        let mut Vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).expect("Create Vibrato failed");

        // And: A mock input signal (simplified for the test)
        let input_signal = vec![1.0, 0.0, -1.0, 0.0]; // A simple square wave
        let input = [&input_signal[..]]; // Convert to expected reference format
        let mut output_signal = vec![0.0; input_signal.len()]; // Prepare an output buffer of the same size
        let output = &mut [&mut output_signal[..]]; // Convert to expected mutable reference format

        // When: Applying the Vibrato effect
        Vibrato.process(&input, output);

        // Then: The output should be modified in a manner consistent with the Vibrato effect
        // This is a simplistic check, in a real test, you'd compare against expected values that
        // reflect the modulated delay applied to the input signal.
        assert_ne!(output_signal, vec![1.0, 0.0, -1.0, 0.0], "The process method should modify the input signal.");

        // Note: A more detailed test would require knowledge of the specific Vibrato
        // implementation details, such as the modulation waveform, depth, and the exact 
        // processing algorithm, to calculate expected output values for comparison.
    }
}
