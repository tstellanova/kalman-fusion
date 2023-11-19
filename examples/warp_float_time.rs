use kalman_fusion::{kalman_update_float, KalmanState};

use chrono::Utc;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Distribution;

/// Simulate multiple clocks providing drifting time measurements
/// of monotonically increasing timestamps,
/// and feed these estimates into a single Kalman filter.
/// Compare the fused estimate against the "true" clock,
/// which is just a monotonically increasing counter
fn main() {
    const NUM_SENSORS: usize = 8;
    const MAX_TIME_STEPS: u32 = 1_000;

    let start_now = Utc::now();
    let base_start_timestamp: i64 = start_now.timestamp();
    let mut my_rng = StdRng::from_entropy();

    for trial in 1..=25 {
        let normal_dist = rand_distr::Normal::<f64>::new(1.0, 1E-4).unwrap();
        let start_timestamp: f64 = (base_start_timestamp + trial * 100) as f64;

        let mut sensor_values: [f64; NUM_SENSORS] = [start_timestamp; NUM_SENSORS];
        let mut internal_sensor_values: [f64; NUM_SENSORS] = [start_timestamp as f64; NUM_SENSORS];

        let mut kstate = KalmanState::new_float (
            start_timestamp,
            1E-3,
            1E-6,
           1E-6,
        );

        for _i in 1..=MAX_TIME_STEPS {
            // Add a "fuzzy" increment of one to the monotonically increasing sensed value
            for j in 0..NUM_SENSORS {
                let rand_blip: f64 = normal_dist.sample(&mut my_rng).abs();
                // the internal state of the sensor might evolve along a float continuum
                internal_sensor_values[j] += rand_blip;
                // but the value readable external to the sensor is an integer
                sensor_values[j] = internal_sensor_values[j].round();
                kstate = kalman_update_float(&kstate, sensor_values[j]);
            }
        }

        let true_val = start_timestamp + MAX_TIME_STEPS as f64;
        let estimated_val = kstate.estimate;
        let diff = true_val - estimated_val;
        // println!("true {} est {}", true_val, estimated_val);
        println!(
            "steps: {}  diff: {} uncertainty: {}",
            MAX_TIME_STEPS, diff, kstate.uncertainty
        );
    }
}
