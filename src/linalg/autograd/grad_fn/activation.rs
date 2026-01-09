use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::{Scalar, Tensor};

pub(crate) struct SigmoidGradFn {
    output: Tensor,
}

impl SigmoidGradFn {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl GradFn for SigmoidGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let out_data = self
            .output
            .storage
            .data
            .iter()
            .zip(grad_output.storage.data.iter())
            .map(|(&o, &g)| g * o * (1.0 - o))
            .collect::<Vec<Scalar>>();

        vec![Tensor::new(out_data, self.output.shape())]
    }
}

pub(crate) struct ReLUGradFn {
    mask: Tensor, // saved forward mask
}

impl ReLUGradFn {
    pub fn new(mask: Tensor) -> Self {
        Self { mask }
    }
}

impl GradFn for ReLUGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let out_data = grad_output
            .storage
            .data
            .iter()
            .zip(self.mask.storage.data.iter())
            .map(|(&g, &m)| g * m)
            .collect::<Vec<Scalar>>();

        vec![Tensor::new(out_data, grad_output.shape())]
    }
}
