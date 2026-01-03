use nn_rs::helpers_grad::optimizer::*;
use nn_rs::models_grad::mnist::*;
use nn_rs::nn_grad::models::NeuralNetwork;

fn main() {
    let mnist = MNIST::load_mnist();
    println!(
        "MNIST dataset loaded with {} training images and {} test images",
        mnist.train_images.len(),
        mnist.test_images.len()
    );
    let test_batches = mnist.to_batches(&mnist.test_images, &mnist.test_labels, 100, true);

    let mut net = if std::env::args().any(|arg| arg == "--load") {
        load()
    } else if std::env::args().any(|arg| arg == "--train") {
        train(&mnist, None)
    } else if std::env::args().any(|arg| arg == "--load-train") {
        let net = load();
        train(&mnist, Some(net))
    } else {
        panic!("Please specify an action: --train, --load, or --test");
    };

    let test_accuracy = mnist.test_model(&test_batches, &mut net);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
}

fn train(mnist: &MNIST, net: Option<NeuralNetwork>) -> NeuralNetwork {
    let mut train_batches = mnist.to_batches(&mnist.train_images, &mnist.train_labels, 10, true);
    println!(
        "Training on {} batches of size {}",
        train_batches.len(),
        train_batches[0].images.shape()[0]
    );
    let optimizer: Box<dyn Optimizer> = if std::env::args().any(|arg| arg == "--adam") {
        Box::from(Adam::new(0.1, 0.9, 0.999, 1e-8))
    } else if std::env::args().any(|arg| arg == "--sgd") {
        Box::from(SGD::new(0.1))
    } else {
        panic!("Please specify an optimizer: --adam or --sgd");
    };

    let net = if let Some(mut trained_net) = net {
        mnist.train(&mut train_batches, 1, optimizer, &mut trained_net);
        trained_net
    } else {
        mnist.train_linear_model(&mut train_batches, 1, optimizer)
    };

    net.dump_memory("mnist_output.bin");
    println!("Model trained and memory dumped to mnist_output.bin");

    net
}

fn load() -> NeuralNetwork {
    NeuralNetwork::restore("mnist_output.bin")
}
