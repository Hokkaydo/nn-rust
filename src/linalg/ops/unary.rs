use crate::linalg::autograd::grad_fn::unary::{
    AbsGradFn, ClampGradFn, ExpGradFn, LogGradFn, NegGradFn, PowGradFn,
};
use crate::linalg::tensor::{InternalTensor, Scalar, Storage, Tensor};
use std::cell::RefCell;
use std::ops::Neg;
use std::rc::Rc;

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let result_data = self
            .storage
            .data
            .iter()
            .copied()
            .map(|x| -x)
            .collect::<Vec<Scalar>>();

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(NegGradFn))
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
        let result_data = self
            .storage
            .data
            .iter()
            .map(|&x| x.powf(exponent))
            .collect::<Vec<Scalar>>();

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(PowGradFn::new(self.clone(), exponent)))
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
        let result_data = self
            .storage
            .data
            .iter()
            .map(|&x| x.abs())
            .collect::<Vec<Scalar>>();

        let requires_grad = self.requires_grad;

        let mask = self
            .storage
            .data
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<Scalar>>();
        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(AbsGradFn::new(Tensor::new(mask, &self.shape))))
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

    /// Clamps the tensor values between min and max
    /// # Arguments
    /// * `min` - Minimum value
    /// * `max` - Maximum value
    /// # Returns
    /// A tensor with values clamped between min and max
    pub fn clamp(&self, min: Scalar, max: Scalar) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        let mut mask = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            if self.storage.data[i] < min {
                result_data.push(min);
                mask.push(0.);
            } else if self.storage.data[i] > max {
                result_data.push(max);
                mask.push(0.);
            } else {
                result_data.push(self.storage.data[i]);
                mask.push(1.);
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
                Some(Rc::new(ClampGradFn::new(Tensor::new(mask, &self.shape))))
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

    /// Computes the natural logarithm of the tensor
    /// # Returns
    /// A tensor containing the logarithm values
    pub fn log(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].ln());
        }

        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(LogGradFn::new(self.clone())))
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

    pub fn exp(&self) -> Tensor {
        let mut result_data = Vec::with_capacity(self.shape.iter().product());
        for i in 0..self.storage.data.len() {
            result_data.push(self.storage.data[i].exp());
        }

        let requires_grad = self.requires_grad;

        let mut out: Tensor = InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None,
            parents: if requires_grad {
                vec![self.clone()]
            } else {
                Vec::new()
            },
            requires_grad,
        }
        .into();

        if requires_grad {
            out.set_grad_metadata(Rc::new(ExpGradFn::new(out.clone())), vec![self.clone()]);
        }
        out
    }

    pub fn sign(&self) -> Tensor {
        let result_data = self
            .storage
            .data
            .iter()
            .map(|x| x.signum())
            .collect::<Vec<_>>();
        let requires_grad = self.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None, // Gradient function for sign not implemented
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
