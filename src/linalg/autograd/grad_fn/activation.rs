use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::Tensor;

pub(crate) struct SigmoidGradFn;

impl GradFn for SigmoidGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_input = grad_output * &(output * &(&Tensor::ones(output.shape()) - output));
        vec![grad_input]
    }
}

/// Mask as argument
pub(crate) struct ReLUGradFn(pub Tensor);

impl GradFn for ReLUGradFn {
    fn apply(&self, _output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_input = grad_output * &self.0;
        vec![grad_input]
    }
}
