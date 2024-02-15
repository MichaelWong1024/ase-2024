pub struct CombFilter {
    filter_type: FilterType,
    gain: f32,
    max_delay_secs: f32,
    sample_rate_hz: f32,
    max_delay_samples: usize,
    num_channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    FIR,
    IIR,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterParam {
    Gain,
    Delay,
}

#[derive(Debug, Clone)]
pub enum Error {
    InvalidValue { param: FilterParam, value: f32 },
}

impl CombFilter {
    pub fn new(filter_type: FilterType, max_delay_secs: f32, sample_rate_hz: f32, num_channels: usize) -> Self {
        CombFilter {
            filter_type,
            gain: 0.0,
            max_delay_secs,
            sample_rate_hz,
            max_delay_samples: (max_delay_secs * sample_rate_hz) as usize,
            num_channels,
        }
    }

    pub fn reset(&mut self) {
        self.gain = 0.0;
        // Instead of resetting max_delay_secs to 0.0, maintain its value or reset to a default if needed
        self.max_delay_samples = (self.max_delay_secs * self.sample_rate_hz) as usize;
    }

    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) {
        // Pre-allocate delay_line outside of the processing loop to avoid repeated allocations
        let mut delay_line = vec![vec![0.0; self.num_channels]; self.max_delay_samples];
        // Process each sample and channel
        for (i, &input_channel) in input.iter().enumerate() {
            for channel in 0..self.num_channels {
                // Update the output with the current input and the delayed sample multiplied by the gain
                output[i][channel] = input_channel[channel] + self.gain * delay_line[0][channel];
                // Shift the delay line
                for delay_index in 0..self.max_delay_samples - 1 {
                    delay_line[delay_index][channel] = delay_line[delay_index + 1][channel];
                }
                // Update the delay line's last element based on the filter type
                delay_line[self.max_delay_samples - 1][channel] = match self.filter_type {
                    FilterType::FIR => input_channel[channel],
                    FilterType::IIR => output[i][channel],
                };
            }
        }
    }

    pub fn set_param(&mut self, param: FilterParam, value: f32) -> Result<(), Error> {
        match param {
            FilterParam::Gain => {
                // Validate gain value, assuming a valid range is 0.0 to 1.0
                if !(0.0..=1.0).contains(&value) {
                    return Err(Error::InvalidValue { param, value });
                }
                self.gain = value;
            }
            FilterParam::Delay => {
                // Ensure the delay is within a plausible range
                if value < 0.0 || value > self.max_delay_secs {
                    return Err(Error::InvalidValue { param, value });
                }
                self.max_delay_secs = value;
                self.max_delay_samples = (self.max_delay_secs * self.sample_rate_hz) as usize;
            }
        }
        Ok(())
    }

    pub fn get_param(&self, param: FilterParam) -> f32 {
        match param {
            FilterParam::Gain => self.gain,
            FilterParam::Delay => self.max_delay_secs,
        }
    }
}
