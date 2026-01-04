use crate::linalg::autograd::grad_fn::activation::{ReLUGradFn, SigmoidGradFn};
use crate::linalg::tensor_grad::{InternalTensor, Scalar, Storage, Tensor};
use crate::not_implemented_grad_fn;
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    /// Computes the sigmoid of the tensor
    /// # Returns
    /// A tensor containing the sigmoid values
    pub fn sigmoid(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            let val = self.storage.data[i];
            result_data.push(1.0 / (1.0 + (-val).exp()));
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(SigmoidGradFn))
            } else {
                None
            },
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
            exp_sum += (self.storage.data[i] - max_val).exp();
        }
        let log_exp_sum = exp_sum.ln() + max_val;

        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for index in 0..self.storage.data.len() {
            result_data.push(self.storage.data[index] - log_exp_sum);
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("LogSoftmax"),
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
        let mut mask = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            let val = self.storage.data[i];
            if val < 0.0 {
                result_data.push(0.0);
                mask.push(0.0);
            } else {
                result_data.push(val);
                mask.push(1.0);
            }
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(ReLUGradFn(Tensor::new(mask, self.shape()))))
            } else {
                None
            },
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
