use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::Tensor;

pub(crate) struct MeanGradFn {
    pub(crate) axes: Vec<usize>,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) scale: f32,
}

impl MeanGradFn {
    pub fn new(axes: Vec<usize>, input_shape: Vec<usize>) -> Self {
        let scale = axes.iter().map(|&a| input_shape[a] as f32).product();
        Self {
            axes,
            input_shape,
            scale,
        }
    }
}

impl GradFn for MeanGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grad_input = grad_output.clone();

        for &axis in &self.axes {
            grad_input = grad_input.expand_dim(axis, self.input_shape[axis]);
        }

        grad_input = grad_input * (1.0 / self.scale);
        vec![grad_input]
    }
}

pub(crate) struct SumAxisGradFn {
    pub(crate) input_shape: Vec<usize>,
}

impl SumAxisGradFn {
    pub fn new(input_shape: Vec<usize>) -> Self {
        Self { input_shape }
    }
}

impl GradFn for SumAxisGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grad_input = grad_output.clone();

        for (axis, &dim) in self.input_shape.iter().enumerate() {
            if grad_input.shape.get(axis).copied().unwrap_or(1) != dim {
                grad_input = grad_input.expand_dim(axis, dim);
            }
        }

        vec![grad_input]
    }
}

pub(crate) struct SumGradFn {
    pub(crate) input_shape: Vec<usize>,
}

impl SumGradFn {
    pub fn new(input_shape: Vec<usize>) -> Self {
        Self { input_shape }
    }
}

impl GradFn for SumGradFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grad_input = grad_output.clone();

        for (axis, &dim) in self.input_shape.iter().enumerate() {
            grad_input = grad_input.expand_dim(axis, dim);
        }

        vec![grad_input]
    }
}
