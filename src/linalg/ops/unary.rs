use crate::linalg::autograd::grad_fn::{TensorNegTensorFn, TensorPowFn};
use crate::linalg::tensor_grad::{Scalar, Storage, Tensor};
use std::cell::RefCell;
use std::ops::Neg;
use std::rc::Rc;

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(-self.storage.data[i].clone());
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(TensorNegTensorFn {}))
            } else {
                None
            },
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        (&self).neg()
    }
}

impl Tensor {
    /// Raises each element of the tensor to the given exponent
    /// # Arguments
    /// * `exponent` - The exponent to raise each element to
    /// # Returns
    /// A tensor containing the results
    pub fn pow(&self, exponent: Scalar) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].clone().powf(exponent));
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(TensorPowFn { exponent }))
            } else {
                None
            },
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }

    /// Computes the square of the tensor
    /// # Returns
    /// A tensor containing the squared values
    pub fn square(&self) -> Tensor {
        self.pow(2.0)
    }

    /// Computes the square root of the tensor
    /// # Returns
    /// A tensor containing the square root values
    pub fn sqrt(&self) -> Tensor {
        self.pow(0.5)
    }

    /// Computes the absolute value of the tensor
    /// # Returns
    /// A tensor containing the absolute values
    pub fn abs(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].clone().abs());
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for abs not implemented
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }

    /// Clamps the tensor values between min and max
    /// # Arguments
    /// * `min` - Minimum value
    /// * `max` - Maximum value
    /// # Returns
    /// A tensor with values clamped between min and max
    pub fn clamp(&self, min: Scalar, max: Scalar) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            let val = self.storage.data[i].clone();
            let clamped = if val < min {
                min
            } else if val > max {
                max
            } else {
                val
            };
            result_data.push(clamped);
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for clamp not implemented
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }

    /// Computes the natural logarithm of the tensor
    /// # Returns
    /// A tensor containing the logarithm values
    pub fn log(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].clone().ln());
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for log not implemented
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }

    pub fn exp(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].clone().exp());
        }

        let requires_grad = self.requires_grad;

        Tensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for exp not implemented
            parents: if requires_grad {
                vec![Rc::new(self.clone())]
            } else {
                Vec::new()
            },
            requires_grad,
        }
    }
}
