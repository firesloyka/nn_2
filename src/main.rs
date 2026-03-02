use ndarray::{Array1, array};
use rand::Rng;

fn perceptron (inputs: &Array1<f64>, weights: &Array1<f64>, bias: f64) -> f64 {
    let sum = weights.dot(inputs)+bias;
    if sum > 0.0 {1.0} else {0.0}
}

fn main() {
 let mut rng = rand::thread_rng();
 let weights = array![rng.r#gen(), rng.r#gen()];
 let bias = rng.r#gen();
 let inputs = array![1.0, 0.0];
 let output = perceptron(&inputs, &weights, bias);
 println!("Output: {}", output);
}
