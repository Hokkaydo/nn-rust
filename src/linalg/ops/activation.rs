use crate::linalg::tensor_grad::{InternalTensor, Scalar, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    /// Computes the sigmoid of the tensor
    /// # Returns
    /// A tensor containing the sigmoid values
    pub fn sigmoid(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            let val = self.storage.data[i].clone();
            result_data.push(1.0 / (1.0 + (-val).exp()));
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for sigmoid not implemented
            parents: if requires_grad {
                vec![self.clone()]
            } else {
                Vec::new()
            },
            requires_grad,
        }
        .into()
    }

    /// Computes the softmax of the tensor
    /// # Returns
    /// A tensor containing the softmax values
    pub fn softmax(&self) -> Tensor {
        let max = self.max();
        let exp_data = (self - &max).exp();
        let sum_exp = exp_data.sum();
        &exp_data / &sum_exp
    }

    /// Computes the log-softmax of the tensor
    /// # Returns
    /// A tensor containing the log-softmax values
    pub fn log_softmax(&self) -> Tensor {
        let max_val = self
            .storage
            .data
            .iter()
            .cloned()
            .fold(Scalar::NEG_INFINITY, Scalar::max);
        let mut exp_sum = 0.0;
        for i in 0..self.storage.data.len() {
            exp_sum += (self.storage.data[i].clone() - max_val).exp();
        }
        let log_exp_sum = exp_sum.ln() + max_val;

        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for index in 0..self.storage.data.len() {
            result_data.push(self.storage.data[index].clone() - log_exp_sum);
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for log_softmax not implemented
            parents: if requires_grad {
                vec![self.clone()]
            } else {
                Vec::new()
            },
            requires_grad,
        }
        .into()
    }

    pub fn relu(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            let val = self.storage.data[i].clone();
            result_data.push(if val < 0.0 { 0.0 } else { val });
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for relu not implemented
            parents: if requires_grad {
                vec![self.clone()]
            } else {
                Vec::new()
            },
            requires_grad,
        }
        .into()
    }
}
