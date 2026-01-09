use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::{Scalar, Tensor};

pub(crate) struct NegGradFn;

impl GradFn for NegGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let data = grad_output
            .storage
            .data
            .iter()
            .map(|&g| -g)
            .collect::<Vec<Scalar>>();

        vec![Tensor::new(data, grad_output.shape())]
    }
}

pub(crate) struct PowGradFn {
    base: Tensor,
    exponent: Scalar,
}

impl PowGradFn {
    pub fn new(base: Tensor, exponent: Scalar) -> Self {
        Self { base, exponent }
    }
}

impl GradFn for PowGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = self.base.storage.data.len();
        let mut grad_input = vec![0.0; n];

        for i in 0..n {
            grad_input[i] = grad_output.storage.data[i]
                * self.exponent
                * self.base.storage.data[i].powf(self.exponent - 1.0);
        }

        vec![Tensor::new(grad_input, self.base.shape())]
    }
}

pub(crate) struct AbsGradFn {
    sign: Tensor, // {-1, 0, 1}
}

impl AbsGradFn {
    pub fn new(sign: Tensor) -> Self {
        Self { sign }
    }
}

impl GradFn for AbsGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = grad_output.storage.data.len();
        let mut grad_input = vec![0.0; n];

        for i in 0..n {
            grad_input[i] = grad_output.storage.data[i] * self.sign.storage.data[i];
        }

        vec![Tensor::new(grad_input, grad_output.shape())]
    }
}

pub(crate) struct ClampGradFn {
    mask: Tensor,
}

impl ClampGradFn {
    pub fn new(mask: Tensor) -> Self {
        Self { mask }
    }
}

impl GradFn for ClampGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = grad_output.storage.data.len();
        let mut grad_input = vec![0.0; n];

        for i in 0..n {
            grad_input[i] = grad_output.storage.data[i] * self.mask.storage.data[i];
        }

        vec![Tensor::new(grad_input, grad_output.shape())]
    }
}

pub(crate) struct LogGradFn {
    input: Tensor,
}

impl LogGradFn {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradFn for LogGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = self.input.storage.data.len();
        let mut grad_input = vec![0.0; n];

        for i in 0..n {
            grad_input[i] = grad_output.storage.data[i] / self.input.storage.data[i];
        }

        vec![Tensor::new(grad_input, self.input.shape())]
    }
}

pub(crate) struct ExpGradFn {
    output: Tensor,
}

impl ExpGradFn {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl GradFn for ExpGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = self.output.storage.data.len();
        let mut grad_input = vec![0.0; n];

        for i in 0..n {
            grad_input[i] = grad_output.storage.data[i] * self.output.storage.data[i];
        }

        vec![Tensor::new(grad_input, self.output.shape())]
    }
}
