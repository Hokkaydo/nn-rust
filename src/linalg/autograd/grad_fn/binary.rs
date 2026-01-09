use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::{Scalar, Tensor};

/// Gradient for element-wise addition
pub(crate) struct AddGradFn {
    parents: Vec<Tensor>,
}

impl AddGradFn {
    pub fn new(parents: Vec<Tensor>) -> Self {
        Self { parents }
    }
}

impl GradFn for AddGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        self.parents
            .iter()
            .map(|parent| grad_output.sum_to_shape(parent.shape()))
            .collect()
    }
}

/// Gradient for subtraction, optionally for left and/or right
pub(crate) struct SubGradFn {
    grad_left: bool,
    grad_right: bool,
    parents: Vec<Tensor>,
}

impl SubGradFn {
    pub fn new(grad_left: bool, grad_right: bool, parents: Vec<Tensor>) -> Self {
        Self {
            grad_left,
            grad_right,
            parents,
        }
    }
}

impl GradFn for SubGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();
        let mut i = 0;
        if self.grad_left {
            let shape = self.parents[i].shape();
            grads.push(grad_output.sum_to_shape(shape));
            i += 1;
        }
        if self.grad_right {
            let shape = self.parents[i].shape();
            grads.push(-grad_output.sum_to_shape(shape));
        }
        grads
    }
}

/// Gradient for element-wise multiplication (supports optional scalar)
pub(crate) struct EWSMultGradFn {
    a: Tensor,
    b: Tensor,
}

impl EWSMultGradFn {
    pub fn new(a: Tensor, b: Option<Scalar>, parent_b: Option<Tensor>) -> Self {
        let b_tensor = b
            .map(Tensor::from_scalar)
            .or(parent_b)
            .expect("Either scalar or tensor must be provided");
        Self { a, b: b_tensor }
    }
}

impl GradFn for EWSMultGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();
        if self.a.requires_grad {
            grads.push((grad_output * &self.b).sum_to_shape(self.a.shape()));
        }
        if self.b.requires_grad {
            grads.push((grad_output * &self.a).sum_to_shape(self.b.shape()));
        }
        grads
    }
}

/// Gradient for division (supports optional scalars)
pub(crate) struct DivGradFn {
    a: Tensor,
    b: Tensor,
}

impl DivGradFn {
    pub fn new(
        a: Option<Scalar>,
        b: Option<Scalar>,
        parent_a: Option<Tensor>,
        parent_b: Option<Tensor>,
    ) -> Self {
        let a_tensor = a
            .map(Tensor::from_scalar)
            .or(parent_a)
            .expect("Either scalar or parent tensor a must be provided");
        let b_tensor = b
            .map(Tensor::from_scalar)
            .or(parent_b)
            .expect("Either scalar or parent tensor b must be provided");
        Self {
            a: a_tensor,
            b: b_tensor,
        }
    }
}

impl GradFn for DivGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();
        if self.a.requires_grad {
            grads.push(grad_output.sum_to_shape(self.a.shape()));
        }
        if self.b.requires_grad {
            grads.push((-grad_output).sum_to_shape(self.b.shape()));
        }
        grads
    }
}
