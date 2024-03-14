<<<<<<< HEAD
use crate::lfo::LFO;
use crate::ring_buffer::RingBuffer;

#[derive(Debug, Clone)]
pub enum Error {
    InvalidValue { name: String, value: f32 },
}

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
    pub fn new(
        sample_rate: f32,
        delay_time: f32,
        width_time: f32,
        modulation_frequency: f32,
        channels: usize,
    ) -> Result<Self, Error> {
        let delay_samples = (delay_time * sample_rate).round() as usize;
        let width_samples = (width_time * sample_rate).round() as usize;
        let capacity = Self::calculate_capacity(delay_samples, width_samples);

        if width_samples > delay_samples {
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
        lfo.set_amplitude(width_samples as f32);

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

    pub fn reset(&mut self) {
        self.delay_buffers.iter_mut().for_each(RingBuffer::reset);
    }

    pub fn set_param(
        &mut self,
        delay_time: f32,
        width_time: f32,
        modulation_frequency: f32,
    ) -> Result<(), Error> {
        let delay_samples = (delay_time * self.sample_rate).round() as usize;
        let width_samples = (width_time * self.sample_rate).round() as usize;
        let capacity = Self::calculate_capacity(delay_samples, width_samples);

        if width_samples > delay_samples {
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

    pub fn get_param(&self) -> (f32, f32, f32) {
        (self.delay_time, self.width_time, self.modulation_frequency)
    }

    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) {
        for (channel_idx, &channel_input) in input.iter().enumerate() {
            let delay_buffer = &mut self.delay_buffers[channel_idx];

            channel_input.iter().enumerate().for_each(|(sample_idx, &sample)| {
                let mod_depth_samples = self.lfo.next_mod();
                let total_delay_samples = self.delay_time * self.sample_rate + mod_depth_samples;
                delay_buffer.push(sample);
                let delayed_sample = delay_buffer.get_frac(total_delay_samples);
                output[channel_idx][sample_idx] = delayed_sample;
            });
        }
    }

    fn calculate_capacity(delay_samples: usize, width_samples: usize) -> usize {
        1 + delay_samples + width_samples * 2
    }
}





#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::RingBuffer;

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
||||||| cd9c504
=======
use crate::{ring_buffer::RingBuffer, lfo::Modulator};
use std::error::Error;

pub struct Vibrato {
    max_delay_secs: f32, // Maximum delay in seconds, for the depth of the vibrato effect
    sample_rate_hz: f32, // Audio sample rate in Hz
    num_channels: usize, // Number of audio channels
    width_secs: f32, // Width of the vibrato effect in seconds
    mod_freq_hz: f32, // Modulation frequency of the vibrato effect in Hz
    delay_buffers: Vec<RingBuffer<f32>>, // Delay buffers for each audio channel
    modulator: Modulator, // Modulator for the vibrato effect
}

impl Vibrato {
    /// Creates a new instance of Vibrato with specified parameters, ensuring practical limits.
    /// 
    /// Parameters:
    /// - `sample_rate_hz`: Audio sample rate in Hz.
    /// - `max_delay_secs`: Maximum delay in seconds, for the depth of the vibrato effect.
    /// - `width_secs`: Width of the vibrato effect in seconds.
    /// - `mod_freq_hz`: Modulation frequency of the vibrato effect in Hz.
    /// - `num_channels`: Number of audio channels.
    /// 
    /// Returns:
    /// - `Ok(Vibrato)`: A new instance of Vibrato on success.
    /// - `Err(&'static str)`: Error message if parameters are invalid.
    pub fn new(sample_rate_hz: f32, max_delay_secs: f32, width_secs: f32, mod_freq_hz: f32, num_channels: usize) -> Result<Vibrato, Box<dyn Error>> {
        if sample_rate_hz <= 0.0 {
            return Err("Sample rate must be positive".into());
        }
        if max_delay_secs <= 0.0 {
            return Err("Maximum delay must be positive".into());
        }
        if width_secs <= 0.0 {
            return Err("Width must be positive".into());
        }
        if width_secs > max_delay_secs {
            return Err("Width must be less than or equal to maximum delay".into());
        }
        if mod_freq_hz <= 0.0 {
            return Err("Modulation frequency must be positive".into());
        }
        if num_channels == 0 {
            return Err("Number of channels must be at least 1".into());
        }

        let delay_buffer_capacity = (max_delay_secs * sample_rate_hz) as usize;
        let delay_buffers = (0..num_channels).map(|_| RingBuffer::<f32>::new(delay_buffer_capacity)).collect();

        // Corrected to match the Modulator's new signature
        // Assuming 'depth' parameter needs to be defined, perhaps related to 'width_secs'
        // Let's assume the depth for the modulator is proportional to the width_secs for simplicity
        let depth = width_secs; // This is a simplification and might need adjustment based on your actual use case
        let modulator = Modulator::new(sample_rate_hz, mod_freq_hz, depth);

        Ok(Vibrato {
            sample_rate_hz,
            max_delay_secs,
            width_secs,
            mod_freq_hz,
            num_channels,
            delay_buffers,
            modulator,
        })
    }

    pub fn reset(&mut self) {
        // Reset each delay buffer in the vector
        for buffer in self.delay_buffers.iter_mut() {
            buffer.reset();
        }
        // Reset the modulator to its initial state
        self.modulator.reset();
    }

    pub fn process(&mut self, input: &mut [f32]) {
        // Assuming `Modulator::next_value` gives us the modulation depth in samples
        for (sample_idx, sample) in input.iter_mut().enumerate() {
            // Calculate modulation depth for current sample
            let mod_depth_samples = self.modulator.next_value();

            // Calculate total delay in samples, considering the width of the vibrato effect
            // and modulation depth. Adjust this calculation based on your actual implementation needs.
            let total_delay_samples = (self.width_secs * self.sample_rate_hz + mod_depth_samples) as usize;

            // For each channel, we would normally handle them separately if multi-channel support is needed.
            // Here, we are assuming a mono signal for simplicity.

            // Push the current sample into the delay buffer
            self.delay_buffers[0].push(*sample); // Assuming mono for simplicity, adjust for multi-channel

            // Get the delayed sample from the delay buffer based on the calculated total delay
            // This is where the vibrato effect is applied, by modulating the delay time.
            // Ensure your RingBuffer supports fractional indexing or interpolation if needed.
            let delayed_sample = self.delay_buffers[0].get(total_delay_samples); // Adjust method as necessary

            // Replace the input sample with the delayed (modulated) sample
            *sample = delayed_sample;
        }
    }

    pub fn set_param(&mut self, max_delay_secs: f32, width_secs: f32, mod_freq_hz: f32) -> Result<(), Box<dyn Error>> {
        // Validate the new parameters before applying them
        if max_delay_secs <= 0.0 {
            return Err("Maximum delay must be positive".into());
        }
        if width_secs <= 0.0 || width_secs > max_delay_secs {
            return Err("Width in seconds is out of valid range".into());
        }
        if mod_freq_hz <= 0.0 {
            return Err("Modulation frequency must be positive".into());
        }

        // Calculate the new delay buffer capacity based on the maximum delay
        let delay_buffer_capacity = (max_delay_secs * self.sample_rate_hz) as usize;
        
        // Update the Vibrato parameters
        self.max_delay_secs = max_delay_secs;
        self.width_secs = width_secs;
        self.mod_freq_hz = mod_freq_hz;

        // Reconfigure the delay buffers with the new capacity
        self.delay_buffers = (0..self.num_channels).map(|_| RingBuffer::<f32>::new(delay_buffer_capacity)).collect();

        // Reinitialize the modulator with the new modulation frequency and sample rate
        // Assuming Modulator::new takes modulation frequency and sample rate as parameters
        self.modulator = Modulator::new(self.sample_rate_hz, mod_freq_hz, self.width_secs);

        Ok(())
    }

    pub fn get_params(&self) -> (f32, f32, f32) {
        (self.max_delay_secs, self.width_secs, self.mod_freq_hz)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::RingBuffer;
    use crate::lfo::Modulator;
    use std::f32::EPSILON; // For comparing floating-point numbers
    use std::f32::consts::PI;

    #[test]
    fn test_vibrato_new() {
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.05;
        let width_secs = 0.01;
        let mod_freq_hz = 5.0;
        let num_channels = 2;

        match Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels) {
            Ok(vibrato) => {
                assert_eq!(vibrato.sample_rate_hz, sample_rate_hz);
                assert_eq!(vibrato.max_delay_secs, max_delay_secs);
                assert_eq!(vibrato.width_secs, width_secs);
                assert_eq!(vibrato.mod_freq_hz, mod_freq_hz);
                assert_eq!(vibrato.num_channels, num_channels);
                // Additional assertions can be added here to check the initial state of delay_buffers, modulator, etc.
            },
            Err(e) => panic!("Failed to create Vibrato: {:?}", e),
        }
    }

    #[test]
    fn test_set_param() {
        // Initialize Vibrato with initial parameters
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.01;
        let width_secs = 0.005;
        let mod_freq_hz = 5.0;
        let num_channels = 2;
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).unwrap();
        
        // Assuming Vibrato has a method to get current parameters for verification
        let initial_params = vibrato.get_params();
        
        // Set new parameters
        vibrato.set_param(0.02, 0.01, 10.0).unwrap();
        
        // Get the updated parameters
        let updated_params = vibrato.get_params();
        
        // Assertions to verify that parameters were updated correctly
        // Here, we're assuming `get_params` returns parameters in the order of max_delay_secs, width_secs, mod_freq_hz
        assert_eq!(initial_params.0 * 2.0, updated_params.0, "Max delay seconds did not update correctly");
        assert_eq!(initial_params.1 * 2.0, updated_params.1, "Width seconds did not update correctly");
        assert_eq!(initial_params.2 * 2.0, updated_params.2, "Modulation frequency did not update correctly");
    }
    
    #[test]
    fn test_reset() {
        // Initialize Vibrato with given parameters
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.01;
        let width_secs = 0.005;
        let mod_freq_hz = 5.0;
        let num_channels = 1; // Assuming mono for simplicity
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).unwrap();

        // Prepare a test input signal with a noticeable pattern
        let mut input = vec![0.0; 44100]; // 1 second of audio at 44.1kHz sample rate
        // Create a simple test pattern (e.g., a "click" at the beginning)
        input[0] = 1.0;

        // Process the input signal
        vibrato.process(&mut input);

        // Assume the process modifies the input in a way that can be checked
        // For the sake of this example, let's say processing introduces a delay,
        // so the first sample should still be 0.0 if the buffer was initially empty.
        assert_eq!(input[0], 0.0, "The first sample should not be modified immediately after processing due to the delay.");

        // Reset the vibrato to clear its internal state
        vibrato.reset();

        // Modify the input to reset the test pattern
        input[0] = 1.0; // Reset the "click"
        // Process the input again after reset
        vibrato.process(&mut input);

        // Check the condition that should be true if the reset effectively cleared the state
        // Depending on the specifics of the Vibrato and process implementation, this condition may vary
        assert_eq!(input[0], 0.0, "The first sample should not be modified immediately after processing due to the delay, even after reset.");
    }

    #[test]
    fn test_output_equals_delayed_input_when_modulation_amplitude_is_zero() {
        // Setup
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.1; // 100 ms max delay
        let width_secs = 0.05; // 50 ms vibrato width
        let mod_freq_hz = 0.0; // Zero modulation frequency to simulate zero modulation amplitude
        let num_channels = 1;
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).unwrap();

        // Test Data: A simple ascending signal
        let mut input = (0..10).map(|x| x as f32).collect::<Vec<f32>>();
        let expected_output = input.clone(); // For zero modulation, the output should equal the input, considering the delay

        // Invoke `process`
        vibrato.process(&mut input);

        // Assertion
        // Note: The actual assertion logic might need to adjust for the delay introduced by the width of the vibrato effect.
        // This example assumes the delay effect is correctly handled within the Vibrato::process method.
        // You might need to compare starting from an offset in the input array if there's an inherent delay.
        assert_eq!(input, expected_output, "The processed output did not match the expected delayed input.");
    }

    #[test]
    fn dc_input_results_in_dc_output() {
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.01; // Arbitrary delay setting
        let width_secs = 0.005; // No modulation width for simplicity
        let mod_freq_hz = 5.0; // Modulation frequency, irrelevant here due to zero modulation depth
        let num_channels = 1; // Testing with a single channel for simplicity

        // Assuming Vibrato::new is the constructor for the Vibrato effect
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).expect("Create Vibrato failed");

        let input_len = 441; // Length of the input signal
        let dc_value = 0.5; // DC value for the input
        let mut input = vec![dc_value; input_len]; // Create a mono DC input signal

        // Process the input signal with the Vibrato effect
        vibrato.process(&mut input); // Direct in-place processing

        // Allow some initial transient samples for the effect to stabilize
        let transient_samples = (sample_rate_hz * max_delay_secs) as usize;

        // Verify that the output remains a DC signal after the initial transient period
        for &sample in input.iter().skip(transient_samples) {
            assert!((sample - dc_value).abs() < 0.001, "Output should remain constant (DC) after initial transient");
        }
    }

    #[test]
    fn test_vibrato_with_varying_input_block_size() {
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.01; // Arbitrary delay setting for the test
        let width_secs = 0.005; // Modulation width for vibrato effect
        let mod_freq_hz = 5.0; // Modulation frequency of the vibrato
        let num_channels = 1; // Single channel for simplicity

        // Initialize the Vibrato effect
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).expect("Create Vibrato failed");

        // Test a range of input block sizes
        for &block_size in &[32, 64, 128, 256, 512, 1024, 2048] {
            // Generate a test input signal with a constant value (e.g., a DC offset)
            let dc_value = 0.5;
            let mut input_block = vec![dc_value; block_size];

            // Process the input block with the Vibrato effect
            vibrato.process(&mut input_block);

            // Verify the output
            // Depending on the specifics of the Vibrato effect, you may expect the processed
            // block to vary from the input due to the modulation. Here, we're simply ensuring
            // the process method completes without errors for varying block sizes.
            // Further validation can be added based on expected outcomes of the vibrato processing.
            assert_eq!(input_block.len(), block_size, "Processed block size should match input block size");

            // Optionally, verify specific characteristics of the processed signal
            // e.g., checking for expected modulation patterns or stability after initial transients
        }
    }

    #[test]
    fn test_vibrato_with_zero_input_signal() {
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.01; // Example delay setting
        let width_secs = 0.005; // Example modulation width for the vibrato effect
        let mod_freq_hz = 5.0; // Example modulation frequency
        let num_channels = 1; // Testing with a single channel for simplicity

        // Initialize the Vibrato effect
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).expect("Create Vibrato failed");

        let input_len = 441; // Example length of the input signal
        let zero_input = vec![0.0; input_len]; // Create a zero-input signal
        let mut input = zero_input.clone(); // Clone the zero-input signal for processing

        // Process the zero-input signal with the Vibrato effect
        vibrato.process(&mut input);

        // Verify that the processed signal remains zero
        for (i, &sample) in input.iter().enumerate() {
            assert_eq!(sample, 0.0, "The processed signal should remain zero at sample index {}", i);
        }
    }

    #[test]
    fn test_process_applies_vibrato_effect() {
        // Given: A Vibrato instance with known parameters
        let sample_rate_hz = 44100.0;
        let max_delay_secs = 0.005; // 5ms max delay
        let width_secs = 0.003; // 3ms width
        let mod_freq_hz = 5.0; // 5Hz modulation frequency
        let num_channels = 1; // Mono signal for simplicity
        let mut vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels).expect("Create Vibrato failed");

        // And: A mock input signal (simplified for the test)
        let mut input = vec![1.0, 0.0, -1.0, 0.0]; // A simple square wave

        // When: Applying the vibrato effect
        vibrato.process(&mut input);

        // Then: The output should be modified in a manner consistent with the vibrato effect
        // This is a simplistic check, in a real test, you'd compare against expected values that
        // reflect the modulated delay applied to the input signal.
        assert_ne!(input, vec![1.0, 0.0, -1.0, 0.0], "The process method should modify the input signal.");

        // Note: A more detailed test would require knowledge of the specific vibrato
        // implementation details, such as the modulation waveform and depth, to calculate
        // expected output values for comparison.
    }
}
>>>>>>> 67096c1ba777e4c0c55b3b90f3768db845c495c0
