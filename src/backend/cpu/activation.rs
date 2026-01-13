use crate::backend::backend::ActivationOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl ActivationOps<Self> for CPUBackend {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_data: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        let result_data: Vec<f32> = exp_data.iter().map(|&x| x / sum_exp).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_data: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        let log_sum_exp = sum_exp.ln();
        let result_data: Vec<f32> = data.iter().map(|&x| x - max_val - log_sum_exp).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(result_data, tensor.shape())
    }
}
