use crate::helpers::metrics::*;
use crate::helpers::optimizer::gradient_descent;
use crate::helpers::stopper::*;
use crate::linalg::tensor::Tensor;
use crate::nn::{Dumpable, activation::*, linear::Linear, memory::Memory, models::NeuralNetwork};
use rand::Rng;
use rand::seq::SliceRandom;
use std::io::Write;

#[allow(unused)]
pub fn some(
    inputs: &Tensor,
    targets: &Tensor,
    epochs: usize,
    mut learning_rate: f32,
) -> NeuralNetwork {
    let mut memory = Memory::new();
    let batch_size = inputs.shape[0];
    let input_size = inputs.shape[1];
    let output_size = targets.shape[1];
    let mut net = NeuralNetwork::init(
        vec![
            Box::new(Linear::init(&mut memory, input_size, 16)),
            Box::new(ReLU::new()),
            Box::new(Linear::init(&mut memory, 16, output_size)),
            Box::new(Sigmoid::new()),
        ],
        memory,
    );

    let mut rng = rand::rng();
    let uniform = rand::distr::Uniform::new(-0.1, 0.1).unwrap();

    let mut file = std::fs::File::create("output.csv").expect("Unable to create file");

    let mut plateau = PlateauDetector::new(
        epochs / 100, // patience
        0.0001,       // threshold
    );
    for epoch in 0..epochs {
        let mut data: Vec<f32> = Vec::new();
        let mut shuffled_indices = (0..batch_size).collect::<Vec<usize>>();
        shuffled_indices.shuffle(&mut rng);

        let input = inputs
            .get_levels(&shuffled_indices)
            .map(|x| x + rng.clone().sample(&uniform) as f32);
        let target = targets.get_levels(&shuffled_indices);
        let output = net.forward(input.clone());
        let loss = mse(&target, &output);
        println!("Epoch {}: Loss = {}", epoch, loss);
        writeln!(file, "{},{}", epoch, loss).expect("Unable to write to file");

        if epochs > epochs / 10 && plateau.has_plateaued(loss) {
            learning_rate *= 0.8;
            println!("Learning rate adjusted to {}", learning_rate);
        }
        if learning_rate < 0.0001 {
            println!("Learning rate too low, stopping training.");
            break;
        }

        net.backward(output - target.clone());
        net.apply_gradients(&*gradient_descent(learning_rate));
    }
    for i in 0..batch_size {
        let input = inputs.get_level(i);
        let output = net.forward(input.clone());
        println!("Input: {:?}, Output: {:?}", input, output);
    }
    net
}
