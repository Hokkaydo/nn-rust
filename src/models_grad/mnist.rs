use crate::helpers_grad::metrics::mse;
use crate::helpers_grad::optimizer::Optimizer;
use crate::linalg::tensor_grad::{Scalar, Tensor};
use crate::nn_grad::activation::{ReLU, Sigmoid};
use crate::nn_grad::linear::Linear;
use crate::nn_grad::models::NeuralNetwork;
use rand::seq::SliceRandom;
use std::io::Read;

pub struct MNIST {
    pub train_images: Vec<Vec<u8>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<u8>>,
    pub test_labels: Vec<u8>,
}

pub struct MNISTBatch {
    pub images: Tensor,
    pub labels: Tensor,
}

impl MNIST {
    pub fn load_mnist() -> Self {
        // Load MNIST data from ./mnist/{train, test}.bin files
        // First 4 bytes of the file are the number of images.
        // Next bytes are sequenced as (label, image) = (4 bytes, 28x28 bytes) for each image.

        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_images = Vec::new();
        let mut test_labels = Vec::new();
        let train_file =
            std::fs::File::open("./mnist/train.bin").expect("Failed to open train file");
        let test_file = std::fs::File::open("./mnist/test.bin").expect("Failed to open test file");

        let mut train_reader = std::io::BufReader::new(train_file);
        let mut test_reader = std::io::BufReader::new(test_file);

        let mut num_train_images = [0u8; 4];
        let mut num_test_images = [0u8; 4];

        train_reader
            .read_exact(&mut num_train_images)
            .expect("Failed to read number of train images");
        test_reader
            .read_exact(&mut num_test_images)
            .expect("Failed to read number of test images");

        let num_train = u32::from_be_bytes(num_train_images) as usize;
        let num_test = u32::from_be_bytes(num_test_images) as usize;
        for _ in 0..num_train {
            let mut label = [0u8; 1];
            train_reader
                .read_exact(&mut label)
                .expect("Failed to read train label");
            train_labels.push(label[0]);

            let mut image = vec![0u8; 28 * 28];
            train_reader
                .read_exact(&mut image)
                .expect("Failed to read train image");
            image
                .iter_mut()
                .for_each(|x| *x = (*x as Scalar / 255.0) as u8);
            train_images.push(image);
        }
        for _ in 0..num_test {
            let mut label = [0u8; 1];
            test_reader
                .read_exact(&mut label)
                .expect("Failed to read test label");
            test_labels.push(label[0]);

            let mut image = vec![0u8; 28 * 28];
            test_reader
                .read_exact(&mut image)
                .expect("Failed to read test image");
            image
                .iter_mut()
                .for_each(|x| *x = (*x as Scalar / 255.0) as u8);
            test_images.push(image);
        }
        MNIST {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }

    fn label_to_one_hot(label: u8) -> Vec<Scalar> {
        let mut one_hot = vec![0.0; 10];
        one_hot[label as usize] = 1.0;
        one_hot
    }

    pub fn to_batches(
        &self,
        images: &Vec<Vec<u8>>,
        labels: &Vec<u8>,
        batch_size: usize,
        flat: bool,
    ) -> Vec<MNISTBatch> {
        let mut batches = Vec::new();
        let num_batches = images.len() / batch_size;

        let mut shuffled_indices: Vec<usize> = (0..images.len()).collect();
        shuffled_indices.shuffle(&mut rand::rng());

        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;
            let indices = &shuffled_indices[start..end];

            let images: Vec<Scalar> = indices
                .iter()
                .flat_map(|&idx| images[idx].iter().map(|&x| x as Scalar / 255.0))
                .collect();

            let labels: Vec<Scalar> = indices
                .iter()
                .flat_map(|&idx| Self::label_to_one_hot(labels[idx as usize]))
                .collect();

            let images_tensor = if flat {
                Tensor::new(images, &[batch_size, 28 * 28])
            } else {
                Tensor::new(images, &[batch_size, 28, 28])
            };

            let labels_tensor = Tensor::new(labels, &[batch_size, 10]);

            batches.push(MNISTBatch {
                images: images_tensor,
                labels: labels_tensor,
            });
        }
        batches
    }

    pub fn train_linear_model(
        &self,
        batches: &mut Vec<MNISTBatch>,
        epochs: usize,
        optimizer: Box<dyn Optimizer>,
    ) -> NeuralNetwork {
        let input_size = 28 * 28; // 28x28 pixels
        let output_size = 10; // 10 classes for digits 0-9

        let mut net = NeuralNetwork::init(vec![
            Box::new(Linear::init(input_size, 128)),
            Box::new(ReLU::default()),
            Box::new(Linear::init(128, output_size)),
            Box::new(Sigmoid::default()),
        ]);

        self.train(batches, epochs, optimizer, &mut net);

        net
    }

    pub fn train(
        &self,
        batches: &mut Vec<MNISTBatch>,
        epochs: usize,
        mut optimizer: Box<dyn Optimizer>,
        net: &mut NeuralNetwork,
    ) {
        for epoch in 0..epochs {
            batches.shuffle(&mut rand::rng());
            for (i, batch) in batches.iter().enumerate() {
                let output = net.forward(batch.images.clone());
                let loss = mse(&batch.labels, &output);
                let loss_scalar = loss.as_scalar();
                println!("Epoch {epoch}: Batch {i} Loss = {loss_scalar}");
                loss.backward();
            }
            optimizer.step(net.parameters_mut());
        }
    }

    pub fn test_model(&self, batches: &Vec<MNISTBatch>, net: &mut NeuralNetwork) -> Scalar {
        let mut correct = 0;
        let mut total = 0;

        for batch in batches {
            let output = net.forward(batch.images.clone());
            for (i, &label) in batch.labels.storage.data.iter().enumerate() {
                let predicted = output.storage.data[i].round() as u8;
                if predicted == label as u8 {
                    correct += 1;
                }
                total += 1;
            }
        }
        correct as Scalar / total as Scalar
    }
}
