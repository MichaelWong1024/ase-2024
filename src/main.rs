use std::{fs::File, io::Write};

mod comb_filter;

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn main() {
    show_info();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 6 {
        eprintln!("Usage: {} <input wave filename> <output wave filename> <filter type: iir or fir> [gain=0.5] [delay=0.001]", args[0]);
        std::process::exit(1);
    }

    let filter_type = match args[3].to_lowercase().as_str() {
        "fir" => comb_filter::FilterType::FIR,
        "iir" => comb_filter::FilterType::IIR,
        _ => {
            eprintln!("<filter type> either 'fir' or 'iir'.");
            return;
        }
    };

    let gain = if args.len() == 4 || args.len() == 5 {
        println!("No gain specified, using default gain of 0.5");
        0.5
    } else {
        args[4].parse::<f32>().unwrap_or_else(|_| {
            eprintln!("Invalid gain value, using default of 0.5");
            0.5
        })
    };

    let delay = if args.len() == 4 {
        println!("No delay specified, using default delay of 0.001");
        0.001
    } else {
        args.get(5).map_or(0.001, |d| d.parse().unwrap_or_else(|_| {
            eprintln!("Invalid delay value, using default of 0.001");
            0.001
        }))
    };

    // Open the input wave file with error handling
    let mut reader = match hound::WavReader::open(&args[1]) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!("Failed to open input wave file: {}", e);
            return;
        }
    };

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f32;
    let channels = spec.channels as usize;

    // Read and process samples
    let input = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect::<Vec<_>>()
        .chunks(channels)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    let mut comb_filter = comb_filter::CombFilter::new(filter_type, delay, sample_rate, channels);
    let _ = comb_filter.set_param(comb_filter::FilterParam::Gain, gain);

    // Process audio data
    let mut output = vec![vec![0.0; channels]; input.len()];
    comb_filter.process(
        &input.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
        &mut output.iter_mut().map(|x| x.as_mut_slice()).collect::<Vec<_>>(),
    );

    // Write output to wave file with error handling
    let mut writer = match hound::WavWriter::create(&args[2], hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: spec.bits_per_sample,
        sample_format: spec.sample_format,
    }) {
        Ok(writer) => writer,
        Err(e) => {
            eprintln!("Failed to create output wave file: {}", e);
            return;
        }
    };

    for sample in output.iter().flat_map(|s| s.iter()) {
        if let Err(e) = writer.write_sample((*sample * i16::MAX as f32) as i16) {
            eprintln!("Failed to write sample: {}", e);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use crate::comb_filter;
    use crate::comb_filter::{CombFilter, FilterType, FilterParam};

    // test1: FIR: Output is zero if input freq matches feedforward
    #[test]
    fn test1() {
        let filter_type = comb_filter::FilterType::FIR;
        let sample_rate_hz = 8000_f32;
        let num_channels = 1_usize;

        let gain = 1.0;
        let delay_secs = 0.0025;

        let mut filter = comb_filter::CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
        filter.set_param(comb_filter::FilterParam::Gain, gain);

        let tone_frequency_hz = 200.0;
        let signal_duration_secs = 3.0;
        
        // Calculate the number of samples
        let total_samples = (sample_rate_hz * signal_duration_secs) as usize;
        
        // Generate input signal
        let input_signal: Vec<Vec<f32>> = (0..total_samples).map(|sample_idx| {
            let time_sec = sample_idx as f32 / sample_rate_hz;
            let sample_value = (2.0 * PI * tone_frequency_hz * time_sec).sin();
            vec![sample_value] // Assuming a single channel
        }).collect();

        let mut processed_signal = vec![vec![0.0_f32; num_channels]; total_samples];
        
        // Process the signal
        filter.process(&input_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(), 
                       &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        
        // Assert that the output signal meets expectations
        for output_sample in processed_signal.iter().skip(100) {
            assert!(output_sample[0].abs() < 1.0e-3, "Output not close to zero as expected");
        }
    }

    // test2: IIR: amount of magnitude increase/decrease if input freq matches feedback
    #[test]
    fn test2() {
        let filter_type = comb_filter::FilterType::IIR;
        let sample_rate_hz = 44100_f32;
        let num_channels = 1_usize;

        let gain = 0.5; // Feedback gain
        let delay_secs = 1.0 / 440.0; // Delay set to match a 440 Hz frequency period

        let mut filter = comb_filter::CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
        filter.set_param(comb_filter::FilterParam::Gain, gain);

        let tone_frequency_hz = 440.0; // A frequency that matches the delay
        let signal_duration_secs = 1.0; // Duration of the signal in seconds

        // Calculate the number of samples
        let total_samples = (sample_rate_hz * signal_duration_secs) as usize;

        // Generate input signal
        let input_signal: Vec<Vec<f32>> = (0..total_samples).map(|sample_idx| {
            let time_sec = sample_idx as f32 / sample_rate_hz;
            let sample_value = 0.5 * (2.0 * PI * tone_frequency_hz * time_sec).sin(); // Signal amplitude is 0.5
            vec![sample_value] // Single channel input
        }).collect();

        let mut processed_signal = vec![vec![0.0_f32; num_channels]; total_samples];

        // Process the signal
        filter.process(&input_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(), 
                       &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        
        // Check for expected magnitude increase in the output
        // Note: This check assumes knowledge about the specific behavior of the IIR filter's feedback mechanism
        // and how it affects the signal. The exact condition might vary depending on the implementation details.
        let initial_magnitude = input_signal[total_samples / 2][0].abs();
        let output_magnitude = processed_signal[total_samples / 2][0].abs();

        assert!(output_magnitude > initial_magnitude, "Output magnitude should increase due to feedback");
    }

    // test3: FIR/IIR: correct result for VARYING input block size
    #[test]
    fn test3() {
        let sample_rate_hz = 44100.0;
        let num_channels = 1;
        let gain = 0.5;
        let delay_secs = 0.005; // 5 ms delay
        let block_sizes = [32, 64, 128, 256, 512, 1024, 2048];

        for &block_size in &block_sizes {
            for &filter_type in &[FilterType::FIR, FilterType::IIR] {
                let mut filter = CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
                filter.set_param(FilterParam::Gain, gain).unwrap();

                // Generate a block of input signal - a simple impulse for clarity
                let mut input_signal = vec![0.0; block_size];
                input_signal[0] = 1.0; // Impulse at the beginning

                let mut output_signal = vec![0.0; block_size];
                let input_slices: Vec<&[f32]> = input_signal.chunks(num_channels).collect();
                let mut output_slices: Vec<&mut [f32]> = output_signal.chunks_mut(num_channels).collect();

                filter.process(&input_slices, &mut output_slices);

                // Check the output - specifics depend on expected behavior, here we check for non-zero output at expected delay index
                let expected_delay_index = (sample_rate_hz * delay_secs) as usize;
                if expected_delay_index < block_size {
                    assert!(output_signal[expected_delay_index] != 0.0, "Filter did not produce expected output at delay index for block size {} and filter type {:?}", block_size, filter_type);
                }

                // Additional checks can be added here depending on the expected mathematical properties of the FIR/IIR filter output
            }
        }
    }

    // test4: FIR: correct processing for zero input signal
    #[test]
    fn test4_fir() {
        let filter_type = FilterType::FIR;
        let sample_rate_hz = 44100_f32;
        let num_channels = 1;
        let gain = 0.5;
        let delay_secs = 0.005; // Arbitrary delay
        
        let mut filter = CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
        filter.set_param(FilterParam::Gain, gain);

        let total_samples = 441; // 10ms of audio at 44100Hz
        let zero_input_signal: Vec<Vec<f32>> = vec![vec![0.0; num_channels]; total_samples];

        let mut processed_signal = vec![vec![0.0_f32; num_channels]; total_samples];

        filter.process(&zero_input_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                       &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        
        for output_sample in processed_signal {
            assert_eq!(output_sample[0], 0.0, "FIR filter did not process zero input signal as expected.");
        }
    }

    // test4: IIR: correct processing for zero input signal
    #[test]
    fn test4_iir() {
        let filter_type = FilterType::IIR;
        let sample_rate_hz = 44100_f32;
        let num_channels = 1;
        let gain = 0.5;
        let delay_secs = 0.005; // Arbitrary delay
        
        let mut filter = CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
        filter.set_param(FilterParam::Gain, gain);

        let total_samples = 441; // 10ms of audio at 44100Hz
        let zero_input_signal: Vec<Vec<f32>> = vec![vec![0.0; num_channels]; total_samples];

        let mut processed_signal = vec![vec![0.0_f32; num_channels]; total_samples];

        filter.process(&zero_input_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                       &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        
        // For an IIR filter, the initial few samples might not be zero due to the feedback loop.
        // Therefore, we check if the filter output stabilizes to zero after some initial transient response.
        // This is a simplified test; in real scenarios, you might want to check for a decrease towards zero.
        for output_sample in processed_signal.iter().skip(20) { // Skip initial samples to allow for transient response
            assert_eq!(output_sample[0], 0.0, "IIR filter did not stabilize to zero for zero input signal.");
        }
    }

    // test5 (additional test): FIR/IIR: correct processing for non-zero input signal
    /// Tests the filter's impulse response.
    /// A unit impulse (a single sample of 1 followed by zeros) is input to the filter.
    /// The test verifies that the filter's first output matches the expected gain for FIR,
    /// and checks the behavior for IIR filters over a few samples.
    #[test]
    fn test_impulse_response() {
        let sample_rate_hz = 44100_f32;
        let num_channels = 1;
        let gain = 0.5;
        let delay_secs = 0.001; // Short delay to quickly observe impulse response

        // Test for FIR filter
        {
            let filter_type = FilterType::FIR;
            let mut filter = CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
            filter.set_param(FilterParam::Gain, gain);

            let impulse_signal: Vec<Vec<f32>> = vec![vec![1.0]; 1]; // Unit impulse
            let mut processed_signal = vec![vec![0.0_f32; num_channels]; 1];

            filter.process(&impulse_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                           &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());

            // For an FIR filter, the first output should match the gain applied to the impulse
            assert_eq!(processed_signal[0][0], 1.0, "FIR filter's impulse response did not match expected output.");
        }

        // Test for IIR filter
        {
            let filter_type = FilterType::IIR;
            let mut filter = CombFilter::new(filter_type, delay_secs, sample_rate_hz, num_channels);
            filter.set_param(FilterParam::Gain, gain);

            let impulse_signal: Vec<Vec<f32>> = vec![vec![1.0]; 1]; // Unit impulse
            let mut processed_signal = vec![vec![0.0_f32; num_channels]; 10]; // Check over multiple samples

            filter.process(&impulse_signal.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                           &mut processed_signal.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());

            // For an IIR filter, the output will show the effect of feedback.
            // Here, we simply verify that the filter responds to the impulse, expecting non-zero output initially.
            assert_ne!(processed_signal[0][0], 0.0, "IIR filter did not respond to impulse as expected.");
            // Further checks could analyze the decay of the impulse response according to the filter's design.
        }
    }

}