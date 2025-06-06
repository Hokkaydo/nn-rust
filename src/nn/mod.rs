pub mod activation;
pub mod linear;
pub mod memory;
pub mod models;

use crate::linalg::tensor::Tensor;
use crate::nn::memory::Memory;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub trait Layer: Dumpable {
    fn forward(&mut self, mem: &mut Memory, input: &Tensor) -> Tensor;
    fn backward(&mut self, mem: &mut Memory, grad_output: &Tensor) -> Tensor;
    /**
    The update function takes the current parameter tensor and its gradient tensor, and returns the updated parameter tensor.
    */
    fn apply_gradients(
        &mut self,
        _mem: &mut Memory,
        _update_function: &dyn Fn(&Tensor, &Tensor) -> Tensor,
    ) {
    }
}

pub trait Dumpable {
    fn new() -> Self
    where
        Self: Sized;
    fn dump(&self, _mem: &Memory, _file: &mut BufWriter<File>) {}
    fn restore(&mut self, _mem: &mut Memory, _file: &mut BufReader<File>) {}
    fn type_id(&self) -> &'static str;
}

impl Tensor {
    pub fn log_softmax(&self) -> Tensor {
        let exp_data: Vec<f32> = self.data.iter().map(|&x| x.exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        let log_softmax_data: Vec<f32> = exp_data.iter().map(|&x| (x / sum_exp).ln()).collect();
        Tensor::new(log_softmax_data, self.shape.clone())
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Computes the Negative Log Likelihood (NLL) loss.
    pub fn nll_loss(&self, target: &Tensor) -> f32 {
        let shape = &self.shape;
        assert_eq!(shape.len(), 2, "Expected 2D tensor for input");
        assert_eq!(
            shape[0], target.shape[0],
            "Input and target batch sizes must match"
        );
        assert_eq!(
            shape[1], target.shape[1],
            "Input and target must have the same number of classes"
        );
        let mut loss = 0.0;
        for (i, &value) in self.data.iter().enumerate() {
            let target_index = target.data[i];
            if target_index == 1.0 {
                loss -= value.ln();
            }
        }
        loss / shape[0] as f32
    }
}
