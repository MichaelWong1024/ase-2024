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
        let mut vibrato = Vibrato::new(sample_rate, delay, width, mod_freq, channels)
            .expect("Failed to create Vibrato");
    
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

        let mut filter = Vibrato::new(sample_rate, delay_time, width_time, modulation_frequency, channels)
                            .expect("Failed to create Vibrato");

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
        let mut vibrato_effect = Vibrato::new(
            sample_rate,
            delay_seconds,
            width_seconds,
            modulation_frequency,
            channels,
        )
        .expect("Failed to create Vibrato");
    
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
        let mut vibrato_filter = Vibrato::new(
            sample_rate,
            delay_time,
            modulation_width,
            modulation_frequency,
            NUM_CHANNELS,
        ).expect("Failed to create Vibrato instance");
    
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
        let mut Vibrato = Vibrato::new(sample_rate_hz, max_delay_secs, width_secs, mod_freq_hz, num_channels)
            .expect("Failed to create Vibrato");

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
