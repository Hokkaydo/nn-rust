use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::{Scalar, Tensor};

pub(crate) struct AddGradFn;
impl GradFn for AddGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        output
            .parents
            .iter()
            .map(|parent| grad_output.sum_to_shape(parent.shape()))
            .collect()
    }
}
pub(crate) struct SubGradFn(pub(crate) bool, pub(crate) bool);

impl GradFn for SubGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();

        let mut i = 0;
        if self.0 {
            let shape = output.parents[i].shape();
            grads.push(grad_output.sum_to_shape(shape));
            i += 1;
        }
        if self.1 {
            let shape = output.parents[i].shape();
            grads.push(-grad_output.sum_to_shape(shape));
        }
        grads
    }
}

// If the multiplication involves a scalar, only one gradient needs to be computed
pub(crate) struct EWSMultGradFn(pub(crate) Option<Scalar>);

impl GradFn for EWSMultGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();
        let a = &output.parents[0];
        let b = &self
            .0
            .map(Tensor::from_scalar)
            .unwrap_or_else(|| output.parents[1].clone());

        if a.requires_grad {
            let grad_a = (grad_output * b).sum_to_shape(a.shape());
            grads.push(grad_a);
        }
        if b.requires_grad {
            let grad_b = (grad_output * a).sum_to_shape(b.shape());
            grads.push(grad_b);
        }
        grads
    }
}

// If the division involves a scalar, only one gradient needs to be computed
pub(crate) struct DivGradFn(pub(crate) Option<Scalar>, pub(crate) Option<Scalar>);

impl GradFn for DivGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();
        let a = &self
            .0
            .map(Tensor::from_scalar)
            .unwrap_or_else(|| output.parents[0].clone());
        let b = &self
            .1
            .map(Tensor::from_scalar)
            .unwrap_or_else(|| output.parents[1].clone());
        if a.requires_grad {
            let shape = a.shape();
            let grad_a = grad_output.sum_to_shape(shape);
            grads.push(grad_a);
        }
        if b.requires_grad {
            let shape = b.shape();
            let neg_grad_b = -grad_output.sum_to_shape(shape);
            grads.push(neg_grad_b);
        }
        grads
    }
}
