
use num_traits::float::Float;

/// Holds state for a simple Kalman filter in a single variable.
/// This can be used to fuse inputs from multiple identical "sensors".
#[derive(Debug, Clone, Copy)]
pub struct KalmanState<T> {
  pub estimate: T,/// Estimated value of the variable
  pub uncertainty: T,/// Calculated total uncertainty in the estimate
  measurement_variance: T,  // Uncertainty in the measurement itself
  process_variance: T,      // Error introduced by uncertainty in the process (model)
}

impl<T> KalmanState<T>
  where T: Float
{
  pub fn new_float(
    estimate: T,
    uncertainty: T,
    measurement_variance: T,
    process_variance: T) -> KalmanState<T>
  {
    KalmanState {
      estimate,
      uncertainty: uncertainty.abs(),
      measurement_variance: measurement_variance.abs(),
      process_variance: process_variance.abs(),
    }
  }
}



/// Kalman update function (fold function) for Float types
pub fn kalman_update_float<T>(state: &KalmanState<T>, observation: T) -> KalmanState<T>
  where
    T: Float,
{
  // Kalman gain
  let kalman_gain = state.uncertainty / (state.uncertainty + state.measurement_variance);

  // Update estimate
  let new_estimate = state.estimate + kalman_gain * (observation - state.estimate);

  // Update uncertainty
  let mut new_uncertainty = (T::one() - kalman_gain) * state.uncertainty;
  // adjust for process variance (normally done in a "predict" step
  new_uncertainty = new_uncertainty + state.process_variance;

  KalmanState::new_float(new_estimate, new_uncertainty,
                         state.measurement_variance,state.process_variance)

}

use fixed::traits::{ Fixed };

impl<T> KalmanState<T>
  where T: Fixed
{
  pub fn new_fixed (
    estimate: T,
    uncertainty: T,
    measurement_variance: T,
    process_variance: T) -> KalmanState<T>
  {
    if T::IS_SIGNED {
      KalmanState {
        estimate,
        uncertainty:
          if uncertainty < 0 { T::ZERO - uncertainty } else { uncertainty },

        measurement_variance:
         if measurement_variance < 0 { T::ZERO - measurement_variance } else { measurement_variance },

        process_variance:
          if process_variance < 0 { T::ZERO - process_variance } else { process_variance },
      }
    }
    else {
      KalmanState {
        estimate,
        uncertainty,
        measurement_variance,
        process_variance,
      }
    }
  }
}



/// Kalman update function (fold function) for Fixed types
pub fn kalman_update_fixed<T>(state: &KalmanState<T>, observation: T) -> KalmanState<T>
  where
    T: Fixed ,
{
  // Kalman gain
  let kalman_gain = state.uncertainty / (state.uncertainty + state.measurement_variance);

  // Update estimate

  let new_estimate = if observation >= state.estimate {
    state.estimate + kalman_gain * (observation - state.estimate)
  } else {
    state.estimate - kalman_gain * (state.estimate - observation)
  };
  // let new_estimate = state.estimate + kalman_gain * (observation - state.estimate);

  // Update uncertainty
  let mut new_uncertainty = (T::TRY_ONE.unwrap() - kalman_gain) * state.uncertainty;
  // adjust for process variance (normally done in a "predict" step
  new_uncertainty = new_uncertainty + state.process_variance;

  // let new_uncertainty =
  //     state.process_variance * state.uncertainty * (T::TRY_ONE.unwrap() - kalman_gain);

  KalmanState::new_fixed(new_estimate, new_uncertainty,
                                state.measurement_variance, state.process_variance)
}

#[cfg(test)]
mod tests {
  use super::*;

  // Helper function for floating-point comparison
  fn assert_near_eq(a: f64, b: f64, tol: f64) {
    assert!(
      (a - b).abs() < tol,
      "assertion failed: `(left !== right)`\n left: `{:?}`,\n right: `{:?}`",
      a,
      b
    );
  }

  fn assert_near_eq_fixed<T: Fixed>(a: T, b: T, tol: T) {
    let diff = if a > b { a - b } else { b - a };
    assert!(
      diff < tol,
      "assertion failed: `(left !== right)`\n left: `{:?}`,\n right: `{:?}`",
      a,
      b
    );
  }

  #[test]
  fn test_kalman_filter_f64() {
    let mut kstate = KalmanState {
      estimate: 0.5f64,
      uncertainty: 0.1,
      measurement_variance: 1E-4,
      process_variance: 1.0,
    };

    for _i in 0..10 {
      kstate = kalman_update_float(&kstate, 1.0);
    }
    println!("{}", kstate.estimate);
    assert_near_eq(kstate.estimate, 1.0, 1E-4);
  }

  #[test]
  fn test_with_monotonic_increasing_f64() {
    let mut kstate = KalmanState::new_float(
      0.0f64,
      1.0,
      1E-6,
      1E-3
    );

    const MAX_ITERATIONS: usize = 1000;
    for i in 1..=MAX_ITERATIONS {
      kstate = kalman_update_float(&kstate, i as f64);
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq(kstate.estimate, MAX_ITERATIONS as f64, 1E-3);
    assert_near_eq(kstate.uncertainty, 0.001, 1E-4);
  }

  use fixed::types::{I16F16, I8F24, U32F32};

  #[test]
  fn test_kalman_filter_i8f24() {
    let mut kstate = KalmanState {
      estimate: I8F24::from_num(0.5),
      uncertainty: I8F24::from_num(0.1),
      measurement_variance: I8F24::from_num(1E-4),
      process_variance: I8F24::from_num(1.0),
    };

    for _i in 0..10 {
      kstate = kalman_update_fixed(&kstate, I8F24::from_num(1.0));
    }
    println!("{}", kstate.estimate);
    assert_near_eq(kstate.estimate.into(), 1.0, 1E-4);
  }

  #[test]
  fn test_with_monotonic_increasing_i8f24() {
    type TestType = I8F24;
    let mut kstate = KalmanState::new_fixed(
      TestType::from_num(0),
      TestType::from_num(1),
      TestType::from_num(1E-6),
      TestType::from_num(1E-3),
    );

    let max_iterations: usize = TestType::MAX.to_num::<u32>() as usize;
    for i in 1..=max_iterations {
      kstate = kalman_update_fixed(&kstate, TestType::from_num(i));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(
      kstate.estimate,
      TestType::from_num(max_iterations),
      TestType::from_num( 2E-3),
    );
    assert_near_eq_fixed(
      kstate.uncertainty,
      TestType::from_num(0.001),
      TestType::from_num(1E-6),
    );
  }

  #[test]
  fn test_with_monotonic_increasing_i16f16() {
    type TestType = I16F16;
    let mut kstate = KalmanState::new_fixed(
      TestType::from_num(1.0),
      TestType::from_num(1.0),
      TestType::from_num(1E-3),
      TestType::from_num(1E-3),
    );

    let max_iterations: usize = TestType::MAX.to_num::<u32>() as usize;
    for i in 1..=max_iterations {
      kstate = kalman_update_fixed(&kstate, TestType::from_num(i));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(
      kstate.estimate,
      TestType::from_num(max_iterations),
      TestType::from_num(0.75),
    );
    assert_near_eq_fixed(
      kstate.uncertainty,
      TestType::from_num(0.001),
      TestType::from_num(1E-3),
    );
  }

  #[test]
  fn test_with_monotonic_increasing_u32f32() {
    type TestType = U32F32;
    let mut kstate = KalmanState::new_fixed (
      TestType::from_num(0),
      TestType::from_num(1),
      TestType::from_num(1E-6),
      TestType::from_num(1E-6),
    );

    let max_iterations: usize = TestType::MAX.to_num::<u32>() as usize;
    let step_size: usize = 1000;
    for i in (1..=max_iterations).step_by(step_size) {
      kstate = kalman_update_fixed(&kstate, TestType::from_num(i));
    }

    // Check the final estimate and error_covariance
    println!("est: {} uncert: {}", kstate.estimate, kstate.uncertainty);
    assert_near_eq_fixed(
      kstate.estimate,
      TestType::from_num(max_iterations),
      TestType::from_num(step_size),
    );
    assert_near_eq_fixed(
      kstate.uncertainty,
      TestType::from_num(2E-6),
      TestType::from_num(1E-6),
    );
  }
}
