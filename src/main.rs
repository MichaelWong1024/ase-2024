use std::{fs::File, io::Write};
use std::env;
use hound;

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn main() {
   show_info();

    // Parse command line arguments
    // First argument is input .wav file, second argument is output text file.
    let args: Vec<String> = std::env::args().collect();

    // TODO: your code here
    if args.len() != 3 {
        eprintln!("Usage: {} <input.wav> <output.txt>", args[0]); 
        std::process::exit(1);
    }

    let input_wave_file = &args[1];
    let output_text_file = &args[2];
    
    // Open the input wave file and determine number of channels
    // TODO: your code here; see `hound::WavReader::open`.
    let mut reader = hound::WavReader::open(input_wave_file).expect("Failed to open WAV file");


    // Read audio data and write it to the output text file (one column per channel)
    // TODO: your code here; we suggest using `hound::WavReader::samples`, `File::create`, and `write!`.
    // Remember to convert the samples to floating point values and respect the number of channels!
    let mut output_file = File::create(output_text_file).expect("Failed to create output file");

    let mut samples_iter = reader.samples::<i16>();

    while let (Some(Ok(left)), Some(Ok(right))) = (samples_iter.next(), samples_iter.next()) {
        let left_f32 = left as f32 / i16::MAX as f32;
        let right_f32 = right as f32 / i16::MAX as f32;
        write!(output_file, "{} {}\n", left_f32, right_f32).expect("Failed to write to output file");
    }

    // for sample in reader.into_samples::<i16>() {
    //     match sample {
    //         Ok(s) => {
    //             let sample_f32 = s as f32 / i16::MAX as f32;
    //             write!(output_file, "{}\n", sample_f32).expect("Failed to write to output file");
    //         },
    //         Err(e) => {
    //             eprintln!("Error: {}", e);
    //             std::process::exit(1);
    //         }
    //     }
    // }
}
