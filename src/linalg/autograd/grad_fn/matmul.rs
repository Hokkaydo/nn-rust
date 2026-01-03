use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::Tensor;

pub(crate) struct MatMulGradFn;

impl GradFn for MatMulGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let a = &output.parents[0];
        let b = &output.parents[1];
        let mut grads = Vec::new();

        if a.requires_grad {
            let grad_a = grad_output.matmul(&b.transpose());
            grads.push(grad_a);
        }
        if b.requires_grad {
            let grad_b = a.transpose().matmul(grad_output);
            grads.push(grad_b);
        }
        grads
    }
}
