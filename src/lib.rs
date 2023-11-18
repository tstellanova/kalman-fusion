use num_traits::{float::{Float}};

#[derive(Debug, Clone, Copy)]
pub struct KalmanState<T> {
  pub estimate: T, // Estimated value of the variable
  pub uncertainty: T, // Uncertainty (variance) in the estimate
  pub measurement_uncertainty: T, // Uncertainty (variance) in the measurement
  pub process_noise: T, //process noise multiplier
}

// Kalman update function (fold function)
pub fn kalman_update_float<T>(state: KalmanState<T>, observation: T) -> KalmanState<T>
  where T: Float
{
  // Kalman gain
  let kalman_gain = state.uncertainty / (state.uncertainty + state.measurement_uncertainty);

  // Update estimate
  let inner = observation - state.estimate;
  let inner2 = kalman_gain * inner;
  let new_estimate = state.estimate + inner2;

  // Updated uncertainty
  let new_uncertainty = state.process_noise * state.uncertainty * (T::one() - kalman_gain);

  KalmanState {
    estimate: new_estimate,
    uncertainty: new_uncertainty,
    measurement_uncertainty: state.measurement_uncertainty,
    process_noise: state.process_noise,
  }
}




use fixed::{
  traits::{Fixed}
};

pub fn kalman_update_fixed<T>(state: KalmanState<T>, observation: T) -> KalmanState<T>
  where T: Fixed
{
  // Kalman gain
  let kalman_gain = state.uncertainty / (state.uncertainty + state.measurement_uncertainty);

  // Update estimate
  let new_estimate = state.estimate + kalman_gain * (observation - state.estimate);

  // Updated uncertainty
  let new_uncertainty = state.process_noise * state.uncertainty * (T::TRY_ONE.unwrap() - kalman_gain);

  KalmanState {
    estimate: new_estimate,
    uncertainty: new_uncertainty,
    measurement_uncertainty: state.measurement_uncertainty,
    process_noise: state.process_noise,
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  // Helper function for floating-point comparison
  fn assert_near_eq(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "assertion failed: `(left !== right)`\n left: `{:?}`,\n right: `{:?}`", a, b);
  }

  fn assert_near_eq_fixed<T: Fixed>(a: T, b: T, tol: T) {
    let diff = if a > b { a - b } else { b - a };
    assert!(diff < tol, "assertion failed: `(left !== right)`\n left: `{:?}`,\n right: `{:?}`", a, b);
  }

  #[test]
  fn test_kalman_filter_f64() {
    let mut kstate = KalmanState {
      estimate: 0.5f64,
      uncertainty: 0.1,
      measurement_uncertainty: 1E-4,
      process_noise: 1.0,
    };

    for _i in 0..10 {
      kstate = kalman_update_float(kstate, 1.0);
    }
    println!("{}",kstate.estimate);
    assert_near_eq(kstate.estimate, 1.0, 1E-4);
  }


  #[test]
  fn test_with_monotonic_increasing_f64() {
    let mut kstate = KalmanState {
      estimate: 0.0f64,
      uncertainty: 1.0,
      measurement_uncertainty: 1E-6,
      process_noise: 2.00,
    };

    const MAX_ITERATIONS:usize = 1000;
    for i in 1..=MAX_ITERATIONS {
      let kalman_gain =  kstate.uncertainty / (kstate.uncertainty + kstate.measurement_uncertainty);
      println!("gain {} uncert: {} munc: {}", kalman_gain, kstate.uncertainty, kstate.measurement_uncertainty);
      kstate = kalman_update_float(kstate, i as f64);
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq(kstate.estimate, MAX_ITERATIONS as f64, 1.01f64);
    assert_near_eq(kstate.uncertainty, 0.001, 0.001);
  }


  use fixed::types::{I16F16, I8F24, U32F32};

  #[test]
  fn test_kalman_filter_i8f24() {
    let mut kstate = KalmanState {
      estimate: I8F24::from_num(0.5),
      uncertainty: I8F24::from_num(0.1),
      measurement_uncertainty: I8F24::from_num(1E-4),
      process_noise: I8F24::from_num(1.0),
    };

    for _i in 0..10 {
      kstate = kalman_update_fixed(kstate, I8F24::from_num(1.0));
    }
    println!("{}",kstate.estimate);
    assert_near_eq(kstate.estimate.into(), 1.0, 1E-4);
  }

  #[test]
  fn test_with_monotonic_increasing_i8f24() {
    type TestType = I8F24;
    let mut kstate = KalmanState {
      estimate: TestType::from_num(0),
      uncertainty: TestType::from_num(1),
      measurement_uncertainty: TestType::from_num(1E-6),
      process_noise: TestType::from_num(2),
    };

    let max_iterations:usize = TestType::MAX.to_num::<u32>() as usize;
    for i in 1..=max_iterations {
      kstate = kalman_update_fixed(kstate, TestType::from_num(i ));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(kstate.estimate, TestType::from_num(max_iterations), TestType::from_num(1.01));
    assert_near_eq_fixed(kstate.uncertainty, TestType::from_num(0.001), TestType::from_num(0.001));
  }

  #[test]
  fn test_with_monotonic_increasing_i16f16() {
    type TestType = I16F16;
    let mut kstate = KalmanState {
      estimate: TestType::from_num(0),
      uncertainty: TestType::from_num(1),
      measurement_uncertainty: TestType::from_num(1E-4),
      process_noise: TestType::from_num(2),
    };

    let max_iterations:usize = TestType::MAX.to_num::<u32>() as usize;
    for i in 1..=max_iterations {
      kstate = kalman_update_fixed(kstate, TestType::from_num(i ));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(kstate.estimate, TestType::from_num(max_iterations), TestType::from_num(1.01));
    assert_near_eq_fixed(kstate.uncertainty, TestType::from_num(0.001), TestType::from_num(0.001));
  }


  #[test]
  fn test_with_monotonic_increasing_u32f32() {
    type TestType = U32F32;
    let mut kstate = KalmanState {
      estimate: TestType::from_num(0),
      uncertainty: TestType::from_num(1),
      measurement_uncertainty: TestType::from_num(1E-6),
      process_noise: TestType::from_num(2.5),
    };

    let max_iterations:usize = TestType::MAX.to_num::<u32>() as usize;
    let step_size:usize = 1000;
    for i in (1..=max_iterations).step_by(step_size){
      kstate = kalman_update_fixed(kstate, TestType::from_num(i ));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(kstate.estimate, TestType::from_num(max_iterations), TestType::from_num(step_size));
    assert_near_eq_fixed(kstate.uncertainty, TestType::from_num(0.001), TestType::from_num(0.001));
  }



}
