use crate::helpers_grad::metrics::*;
use crate::helpers_grad::stopper::*;
use crate::linalg::tensor_grad::{Scalar, Tensor};
use crate::nn_grad::{activation::*, linear::Linear, models::NeuralNetwork};
use rand::Rng;
use rand::seq::SliceRandom;
use std::io::Write;

#[allow(unused)]
pub fn some(
    inputs: &Tensor,
    targets: &Tensor,
    epochs: usize,
    mut learning_rate: Scalar,
) -> NeuralNetwork {
    let batch_size = inputs.shape()[0];
    let input_size = inputs.shape()[1];
    let output_size = targets.shape()[1];
    let mut net = NeuralNetwork::init(vec![
        Box::new(Linear::init(input_size, 16)),
        Box::new(ReLU::default()),
        Box::new(Linear::init(16, output_size)),
        Box::new(Sigmoid::default()),
    ]);

    let mut rng = rand::rng();
    let uniform = rand::distr::Uniform::new(-0.1, 0.1).unwrap();

    let mut file = std::fs::File::create("output.csv").expect("Unable to create file");

    let mut plateau = PlateauDetector::new(
        epochs / 100, // patience
        0.0001,       // threshold
    );
    for epoch in 0..epochs {
        let mut data: Vec<Scalar> = Vec::new();
        let mut shuffled_indices = (0..batch_size).collect::<Vec<usize>>();
        shuffled_indices.shuffle(&mut rng);

        let noise: Vec<Scalar> = (0..batch_size)
            .map(|_| rng.clone().sample(&uniform) as Scalar)
            .collect();
        let noise_tensor = Tensor::new(noise, &[batch_size, 1]);
        let input = inputs.gather(0, &shuffled_indices) + noise_tensor;
        let target = targets.gather(0, &shuffled_indices);
        let output = net.forward(input.clone());
        let loss = mse(&target, &output);
        let loss_scalar = loss.as_scalar().unwrap_or(0.0);
        println!("Epoch {}: Loss = {}", epoch, loss_scalar);
        writeln!(file, "{},{}", epoch, loss_scalar).expect("Unable to write to file");

        if epochs > epochs / 10 && plateau.has_plateaued(loss_scalar) {
            learning_rate *= 0.8;
            println!("Learning rate adjusted to {}", learning_rate);
        }
        if learning_rate < 0.0001 {
            println!("Learning rate too low, stopping training.");
            break;
        }

        net.backward(output - target.clone());
        // net.apply_gradients(&*gradient_descent(learning_rate));
    }
    for i in 0..batch_size {
        let input = inputs.slice(0, i, 1);
        let output = net.forward(input.clone());
        println!("Input: {:?}, Output: {:?}", input, output);
    }
    net
}
