use hound::{WavReader, WavWriter};
use std::env;
use std::path::Path;

mod lfo;
mod ring_buffer;
mod vibrato;
<<<<<<< HEAD
use vibrato::Vibrato;
||||||| cd9c504
=======
mod lfo;
>>>>>>> 67096c1ba777e4c0c55b3b90f3768db845c495c0

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
    let mut vibrato_filter = Vibrato::new(sample_rate, DELAY, WIDTH, MOD_FREQ, num_channels)
        .expect("Create VibratoFilter failed");

    // Prepare the output WAV file
    let mut writer = WavWriter::create(output_path, spec).expect("Create WAV file failed");

    // Improved sample processing
    let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    let num_samples = samples.len() / num_channels;

    // Organize samples by channel and convert to f32
    let mut channel_samples: Vec<Vec<f32>> = vec![Vec::with_capacity(num_samples); num_channels];
    samples.iter().enumerate().for_each(|(i, &sample)| {
        let channel = i % num_channels;
        channel_samples[channel].push(sample as f32 / i16::MAX as f32);
    });

    // Prepare output samples container
    let mut processed_samples: Vec<Vec<f32>> = vec![vec![0.0; num_samples]; num_channels];

    // Process samples through the Vibrato filter
    vibrato_filter.process(
        &channel_samples.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>(),
        &mut processed_samples.iter_mut().map(|v| v.as_mut_slice()).collect::<Vec<&mut [f32]>>(),
    );

    // Interleave and write processed samples back to the WAV file
    for i in 0..num_samples {
        for channel in 0..num_channels {
            let sample = (processed_samples[channel][i] * i16::MAX as f32).round() as i16;
            writer.write_sample(sample).expect("Write sample failed");
        }
    }

    writer.finalize().expect("Finalize WAV file failed");
}
