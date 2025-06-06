use facial_recognition::linalg::tensor::Tensor;
use facial_recognition::models::simple_linear::*;
use facial_recognition::nn::Dumpable;

fn main() {
    /*
    for i in 0..batch_size {
            let mut xs = inputs[selected_inputs[i]].clone();
            for x in xs.iter_mut() {
                *x += rng.sample(&uniform) as f32;
            }
            xs.iter().for_each(|&x| data.push(x));
        }
        let input = Tensor::new(data, vec![batch_size, input_size]);
        let mut targets_data = Vec::new();
        for i in 0..batch_size {
            let ys: Vec<f32> = targets[selected_inputs[i]].clone();
            ys.iter().for_each(|&y| targets_data.push(y));
        }
        let target = Tensor::new(targets_data, vec![batch_size, output_size]);
    */
    let inputs = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let targets = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let inputs = Tensor::new(inputs, vec![4, 10]);
    let targets = Tensor::new(targets, vec![4, 2]);

    // let mut memory = Memory::new();
    // let batch_size = inputs.shape[0];
    // let input_size = inputs.shape[1];
    // let output_size = targets.shape[1];
    // let mut net = NeuralNetwork::new();
    // net.restore_memory("output.bin");
    // println!("Memory restored from output.bin");
    // println!("{:?}", net.forward(inputs.clone()));

    let net = some(&inputs, &targets, 10000, 0.1);
    net.dump_memory("output.bin");
}
