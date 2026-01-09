use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::Tensor;

pub(crate) struct TransposeGradFn;

impl GradFn for TransposeGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.transpose()]
    }
}
