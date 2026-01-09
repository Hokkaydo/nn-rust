use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::Tensor;

pub struct MatMulGradFn {
    pub lhs: Tensor, // normalized A
    pub rhs: Tensor, // normalized B
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
}

impl GradFn for MatMulGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grads = Vec::new();

        // dL/dA
        if self.lhs.requires_grad {
            let mut grad_a = grad_output.matmul(&self.rhs.transpose());
            // only sum if lhs_shape had extra dims (unsqueezed)
            if grad_a.shape() != self.lhs_shape {
                grad_a = grad_a.sum_to_shape(&self.lhs_shape);
            }
            grads.push(grad_a);
        }

        // dL/dB
        if self.rhs.requires_grad {
            let mut grad_b = self.lhs.transpose().matmul(grad_output);
            if grad_b.shape() != self.rhs_shape {
                grad_b = grad_b.sum_to_shape(&self.rhs_shape);
            }
            grads.push(grad_b);
        }

        grads
    }
}
