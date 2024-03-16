use hound::{WavReader, WavWriter};
use std::env;
use std::path::Path;

mod lfo;
mod ring_buffer;
mod vibrato;
use vibrato::Vibrato;

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn main() {
    show_info();

    // Parse command line arguments more efficiently
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input wave filename> <output wave filename>", args[0]);
        return;
    }

    // Simplify path handling
    let input_path = &args[1];
    let output_path = &args[2];

    // Open input WAV file and initialize reader
    let mut reader = WavReader::open(input_path).expect("Open input WAV file failed");
    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as f32;

    // Constants for Vibrato effect
    const DELAY: f32 = 0.1;
    const WIDTH: f32 = 0.1;
    const MOD_FREQ: f32 = 5.0;

    // Initialize the Vibrato filter
    let mut v = Vibrato::new(sample_rate, DELAY, WIDTH, MOD_FREQ, num_channels).expect("Create Vibrato failed");

    // Prepare the output WAV file
    let mut output = WavWriter::create(output_path, spec).expect("Create WAV file failed");

    // Improved sample processing
    let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();

    // Organize samples by channel and convert to f32
    let mut channel_samples: Vec<Vec<f32>> = vec![Vec::with_capacity(samples.len() / num_channels); num_channels];
    samples.iter().enumerate().for_each(|(i, &sample)| {
        let channel = i % num_channels;
        channel_samples[channel].push(sample as f32 / i16::MAX as f32);
    });

    // Prepare output samples container
    let mut processed_samples: Vec<Vec<f32>> = vec![vec![0.0; samples.len() / num_channels]; num_channels];

    // Process samples through the Vibrato filter
    v.process(
        &channel_samples.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>(),
        &mut processed_samples.iter_mut().map(|v| v.as_mut_slice()).collect::<Vec<&mut [f32]>>(),
    );

    // Interleave and write processed samples back to the WAV file
    for i in 0..(samples.len() / num_channels) {
        for channel in 0..num_channels {
            output.write_sample((processed_samples[channel][i] * i16::MAX as f32).round() as i16).expect("Write sample failed");
        }
    }

    output.finalize().expect("Finalize WAV file failed");
}
