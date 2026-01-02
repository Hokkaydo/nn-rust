use crate::linalg::tensor_grad::{Scalar, Tensor};
use crate::nn_grad::{Dumpable, Layer};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn init(n_inputs: usize, n_outputs: usize) -> Self {
        // let fan_avg = (n_inputs + n_outputs) as Scalar / 2.0;
        // let range = rand::distr::Uniform::new(-6.0 / fan_avg.sqrt(), 6.0 / fan_avg.sqrt()).unwrap();
        let range = rand::distr::Uniform::new(
            -6.0 / (n_inputs as Scalar).sqrt(),
            6.0 / (n_inputs as Scalar).sqrt(),
        )
        .unwrap();
        let weights: Vec<Scalar> = rand::rng()
            .sample_iter(range)
            .take(n_inputs * n_outputs)
            .collect();
        let bias: Vec<Scalar> = rand::rng().sample_iter(range).take(n_outputs).collect();

        let weights = Tensor::new(weights, &[n_inputs, n_outputs]);
        let bias = Tensor::new(bias, &[1, n_outputs]);

        Linear { weights, bias }
    }
}

impl Dumpable for Linear {
    fn dump(&self, file: &mut BufWriter<File>) {
        let weights = &self.weights;
        let bias = &self.bias;

        let mut sizes = weights
            .shape
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect::<Vec<u8>>();
        sizes.append(
            &mut bias
                .shape
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );

        file.write_all(&sizes)
            .expect("Unable to write sizes to file");
        file.write_all(
            &weights
                .storage
                .data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Unable to write weights to file");
        file.write_all(
            &bias
                .storage
                .data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Unable to write bias to file");
    }
    fn restore(file: &mut BufReader<File>) -> Box<dyn Dumpable> {
        let mut sizes = [0u8; 32]; // 4 * 8 bytes for 4 usize values
        file.read_exact(&mut sizes)
            .expect("Unable to read sizes from file");

        let weights_shape = (
            usize::from_le_bytes(sizes[0..8].try_into().unwrap()),
            usize::from_le_bytes(sizes[8..16].try_into().unwrap()),
        );
        let bias_shape = (
            usize::from_le_bytes(sizes[16..24].try_into().unwrap()),
            usize::from_le_bytes(sizes[24..32].try_into().unwrap()),
        );

        let weights_size = weights_shape.0 * weights_shape.1;
        let bias_size = bias_shape.1;

        let mut weights_data = vec![0.0; weights_size];
        let mut bias_data = vec![0.0; bias_size];

        file.read_exact(bytemuck::cast_slice_mut(&mut weights_data))
            .expect("Unable to read weights from file");
        file.read_exact(bytemuck::cast_slice_mut(&mut bias_data))
            .expect("Unable to read bias from file");

        let weights = Tensor::new(weights_data, &[weights_shape.0, weights_shape.1]);
        let bias = Tensor::new(bias_data, &[bias_shape.0, bias_shape.1]);

        Box::new(Linear { weights, bias })
    }
    fn type_id() -> &'static str {
        "linear"
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights).broadcast_add(&self.bias)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}
