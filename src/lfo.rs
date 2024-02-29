<<<<<<< HEAD
use crate::ring_buffer::RingBuffer;
use std::f32::consts::PI;

pub struct LFO {
    wave_table: RingBuffer<f32>,
    sample_rate: f32,
    phase: f32,
    frequency: f32,
    amplitude: f32,
}

impl LFO {
    pub fn new(sample_rate: f32, size: usize) -> Self {
        let mut wave_table = RingBuffer::<f32>::new(size);
        // Pre-compute the wave table using a single sine wave cycle
        let wave_length = 2.0 * PI;
        for i in 0..size {
            let phase = (i as f32 / size as f32) * wave_length;
            wave_table.push(phase.sin());
        }

        Self {
            wave_table,
            sample_rate,
            frequency: 0.0, // Default frequency
            amplitude: 1.0, // Default amplitude
            phase: 0.0, // Initial phase
        }
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase % (2.0 * PI); // Ensure phase is within a valid range
    }

    pub fn get_params(&self) -> (f32, f32, f32) {
        (self.frequency, self.amplitude, self.phase)
    }

    pub fn reset(&mut self, size: usize) {
        self.phase = 0.0;
        // Re-initialize the wave table to avoid duplicate logic
        *self = Self::new(self.sample_rate, size);
    }

    pub fn next_mod(&mut self) -> f32 {
        let phase_increment = 2.0 * PI * self.frequency / self.sample_rate;
        self.phase = (self.phase + phase_increment) % (2.0 * PI);

        let normalized_phase = self.phase / (2.0 * PI);
        let table_index = normalized_phase * self.wave_table.capacity() as f32;

        // Assuming `get_frac` interpolates values in the wave table based on a floating-point index
        self.wave_table.get_frac(table_index) * self.amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn test_next_mod() {
        // Setup for LFO with a sine wave, assuming a sample rate of 44100 Hz and a wave table size of 1024
        let mut lfo = LFO::new(44100.0, 1024);
        lfo.set_frequency(1.0); // Set LFO frequency to 1 Hz
        lfo.set_amplitude(1.0); // Set amplitude to 1.0

        // Assuming a step size based on the frequency and sample rate, calculate the expected value.
        // This is a simplified example to check if the next modulated value is as expected at the start.
        let val = lfo.next_mod();

        // Be careful with float comparison
        // The expected value depends on the initial phase and how the LFO is implemented.
        // Here, we're assuming the initial value should be close to 0 since the phase starts at 0.
        // This expectation might need adjustment based on the actual implementation details of get_frac and the initial phase.
        assert!(
            (val - 0.0).abs() < 0.0001,
            "Value was not as expected: {}. Expected approximately: 0.0",
            val
        );
    }

    #[test]
    fn test_next_value_fractional_advance() {
        // Adjusted setup for a realistic fractional advance scenario
        let sample_rate = 44100.0; // Using a common audio sample rate
        let freq_hz = 1.0; // LFO frequency of 1 Hz for simplicity
        let amplitude = 1.0; // Full amplitude for clear measurement
        
        // Initialize the LFO with these parameters
        let size = sample_rate as usize; // Assuming size of the wave table is equivalent to the sample rate for high resolution
        let mut lfo = LFO::new(sample_rate, size);
        lfo.set_frequency(freq_hz);
        lfo.set_amplitude(amplitude);

        // Calculate an expected step size to ensure a fractional advance
        let expected_steps = sample_rate / freq_hz;
        let fractional_advance_step = expected_steps / 4.0; // Choosing a quarter step for clear fractional advance

        // Simulate a fractional advance in the LFO's phase
        // Assuming the LFO's oscillator completes a full cycle per second at 1 Hz,
        // and considering we're advancing by a quarter of the expected steps for a full cycle,
        // we should be at a phase equivalent to PI/2 (90 degrees) into the sine wave, which would be 1.0 in amplitude.
        let expected_value = amplitude; // At PI/2, the sine wave amplitude should be 1.0, given full amplitude setting

        // Advance the LFO's phase manually to simulate fractional advance
        for _ in 0..(fractional_advance_step as usize) {
            lfo.next_mod(); // This advances the phase
        }

        // Get the actual value after the fractional advance
        let actual_value = lfo.next_mod();

        assert!(
            (actual_value - expected_value).abs() < 0.0001,
            "Value was not as expected: {}. Expected approximately: {}",
            actual_value,
            expected_value
        );
    }
}
||||||| cd9c504
=======
use crate::ring_buffer::RingBuffer;
use std::f32::consts::PI;

pub struct Modulator {
    oscillator: RingBuffer<f32>,
    sample_rate: f32,
    current_phase: f32,
    modulation_frequency: f32,
    depth: f32,
}

impl Modulator {
    pub fn new(sample_rate: f32, modulation_frequency: f32, depth: f32) -> Self {
        let size = (sample_rate / modulation_frequency).round() as usize;
        let mut oscillator = RingBuffer::<f32>::new(size);
        for i in 0..size {
            let phase = (i as f32 / size as f32) * 2.0 * PI;
            oscillator.push(phase.sin());
        }

        Self {
            oscillator,
            sample_rate,
            current_phase: 0.0,
            modulation_frequency,
            depth,
        }
    }

    pub fn set_modulation_frequency(&mut self, modulation_frequency: f32) {
        self.modulation_frequency = modulation_frequency;
        // Optionally, you might want to rebuild the oscillator here if the frequency change is significant.
    }

    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth;
    }

    pub fn set_phase(&mut self, phase: f32) {
        self.current_phase = phase;
    }

    pub fn get_params(&self) -> (f32, f32, f32) {
        (self.modulation_frequency, self.depth, self.current_phase)
    }

    pub fn reset(&mut self) {
        self.current_phase = 0.0;
    }

    /// Calculates the step size for the oscillator based on the current modulation frequency.
    pub fn calculate_step_size(&self) -> f32 {
        let table_size = self.oscillator.capacity() as f32;
        let phase_increment = self.modulation_frequency / self.sample_rate;
        phase_increment * table_size
    }

    /// Generates the next modulation value based on the current phase and step size.
    pub fn next_value(&mut self) -> f32 {
        let step_size = self.calculate_step_size();
        let table_size = self.oscillator.capacity() as f32;
        self.current_phase += step_size;
        if self.current_phase >= table_size {
            self.current_phase -= table_size;
        }

        // Assuming RingBuffer.get_frac is a method that interpolates based on a fractional index
        self.oscillator.get_frac(self.current_phase) * self.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_next_value() {
        // Simplified setup for understanding; assumes oscillator is a sine wave for this frequency
        let mut modulator = Modulator::new(2.0, 1.0, 1.0);
        modulator.set_modulation_frequency(1.0);
        modulator.set_depth(1.0);
        let step_size = modulator.calculate_step_size();
        let val = modulator.next_value();
        // Be careful with float comparison
        assert!((val - 0.0).abs() < 0.0001, "Value was not as expected: {}. Expected approximately: 0.0", val);
    }

    #[test]
    fn test_next_value_fractional_advance() {
        // Adjusted setup for a realistic fractional advance scenario
        let sample_rate = 44100.0; // Using a common audio sample rate
        let mod_frequency = 1.0; // Modulation frequency of 1 Hz for simplicity
        let depth = 1.0; // Full depth for clear measurement
        
        // Initialize the modulator with these parameters
        let mut modulator = Modulator::new(sample_rate, mod_frequency, depth);
        modulator.set_modulation_frequency(mod_frequency);
        modulator.set_depth(depth);

        // Calculate an expected step size to ensure a fractional advance
        let expected_steps = sample_rate / mod_frequency;
        let fractional_advance_step = expected_steps / 4.0; // Choosing a quarter step for clear fractional advance

        // Simulate a fractional advance in the oscillator's phase
        // Assuming the modulator's oscillator completes a full cycle per second at 1 Hz,
        // and considering we're advancing by a quarter of the expected steps for a full cycle,
        // we should be at a phase equivalent to PI/2 (90 degrees) into the sine wave, which would be 1.0 in amplitude.
        let expected_value = (PI / 2.0).sin() * depth; // This should ideally be 1.0, but we're using the formula for clarity

        // Advance the modulator's phase manually to simulate fractional advance
        for _ in 0..(fractional_advance_step as usize) {
            modulator.next_value(); // This advances the phase
        }

        // Get the actual value after the fractional advance
        let actual_value = modulator.next_value();

        assert!(
            (actual_value - expected_value).abs() < 0.0001,
            "Value was not as expected: {}. Expected approximately: {}",
            actual_value,
            expected_value
        );
    }
}
>>>>>>> 67096c1ba777e4c0c55b3b90f3768db845c495c0
