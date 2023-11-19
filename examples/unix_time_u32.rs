use kalman_fusion::{kalman_update_fixed, KalmanState};

use chrono::Utc;
use fixed::types::U32F32;
use std::thread::sleep;
use std::time::Duration;

/**
Use U32F32 from the Fixed crate
with the system clock (which should be monotonically increasing)
to iteratively update a Kalman state estimate
*/
fn main() {
    let now = Utc::now();
    let now_timestamp = now.timestamp() as u32;
    type FixedType = U32F32;

    let mut kstate = KalmanState::new_fixed(
        FixedType::from_num(now_timestamp),
        FixedType::from_num(1E-3),
        FixedType::from_num(1E-6),
        FixedType::from_num(1E-6),
    );

    let max_iterations: usize = 100;
    for _i in 1..=max_iterations {
        let now = Utc::now();
        let now_timestamp = now.timestamp() as u32;
        kstate = kalman_update_fixed(&kstate, FixedType::from_num(now_timestamp));
        println!(
            "true: {} est: {} unc: {}",
            now_timestamp,
            kstate.estimate.round(),
            kstate.uncertainty
        );
        let subsec_micros = now.timestamp_subsec_micros();
        let fall_back = Duration::from_micros(subsec_micros.into());
        let wait_duration = Duration::from_secs(2).checked_sub(fall_back).unwrap();
        sleep(wait_duration);
    }
}
