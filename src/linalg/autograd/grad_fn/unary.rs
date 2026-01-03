use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::{Scalar, Tensor};

pub(crate) struct NegGradFn;

impl GradFn for NegGradFn {
    fn apply(&self, _output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        vec![-grad_output.clone()]
    }
}
pub(crate) struct PowGradFn(pub(crate) Scalar);

impl GradFn for PowGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let exponent = self.0;
        let parent = &output.parents[0];
        let grad_input = grad_output * &(parent.pow(exponent - 1.0) * exponent);
        vec![grad_input]
    }
}
pub(crate) struct AbsGradFn;

impl GradFn for AbsGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let parent = &output.parents[0];
        let grad_input = grad_output * &parent.sign();
        vec![grad_input]
    }
}

/// Mask as argument
pub(crate) struct ClampGradFn(pub Tensor);

impl GradFn for ClampGradFn {
    fn apply(&self, _output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let mask = &self.0;
        vec![grad_output * mask]
    }
}

pub(crate) struct LogGradFn;

impl GradFn for LogGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let parent = &output.parents[0];
        let grad_input = grad_output / parent;
        vec![grad_input]
    }
}

pub(crate) struct ExpGradFn;

impl GradFn for ExpGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_input = grad_output * output;
        vec![grad_input]
    }
}
