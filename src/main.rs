use facial_recognition::helpers::optimizer::gradient_descent;
use facial_recognition::models::mnist::*;

fn main() {
    // let inputs = vec![
    //     1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    //     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    //     0.0, 0.0, 0.0, 0.0,
    // ];
    // let targets = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    // let inputs = Tensor::new(inputs, vec![4, 10]);
    // let targets = Tensor::new(targets, vec![4, 2]);

    // let mut memory = Memory::new();
    // let batch_size = inputs.shape[0];
    // let input_size = inputs.shape[1];
    // let output_size = targets.shape[1];
    // let mut net = NeuralNetwork::new();
    // net.restore_memory("output.bin");
    // println!("Memory restored from output.bin");
    // println!("{:?}", net.forward(inputs.clone()));

    // let net = some(&inputs, &targets, 10000, 0.1);
    // net.dump_memory("output.bin");

    let mnist = MNIST::load_mnist();
    println!(
        "MNIST dataset loaded with {} training images and {} test images",
        mnist.train_images.len(),
        mnist.test_images.len()
    );
    let mut train_batches = mnist.to_batches(&mnist.train_images, &mnist.train_labels, 10, true);
    let test_batches = mnist.to_batches(&mnist.test_images, &mnist.test_labels, 100, true);

    let mut net = mnist.train_linear_model(&mut train_batches, 5, gradient_descent(0.1));
    net.dump_memory("mnist_output.bin");
    println!("Model trained and memory dumped to mnist_output.bin");

    let test_accuracy = mnist.test_model(&test_batches, &mut net);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
}
