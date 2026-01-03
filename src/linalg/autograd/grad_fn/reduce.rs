use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::Tensor;

pub(crate) struct MeanGradFn {
    axes: Vec<usize>,
    input_shape: Vec<usize>,
}

impl MeanGradFn {
    pub fn new(axes: Vec<usize>, input_shape: Vec<usize>) -> Self {
        MeanGradFn { axes, input_shape }
    }
}

impl GradFn for MeanGradFn {
    fn apply(&self, _output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let mut grad_input = grad_output.clone();
        for &axis in &self.axes {
            grad_input = grad_input.expand_dim(axis, self.input_shape[axis]);
        }
        let mut scale = 1.0;
        for &axis in &self.axes {
            scale *= self.input_shape[axis] as f32;
        }
        grad_input = grad_input * (1.0 / scale);
        vec![grad_input]
    }
}

pub(crate) struct SumAxisGradFn;

impl GradFn for SumAxisGradFn {
    fn apply(&self, output: &Tensor, grad_output: &Tensor) -> Vec<Tensor> {
        let input_shape = &output.parents[0].shape;
        let mut grad_input = grad_output.clone();
        for (i, &dim) in input_shape.iter().enumerate() {
            if dim != output.shape.get(i).cloned().unwrap_or(1) {
                grad_input = grad_input.expand_dim(i, dim);
            }
        }
        vec![grad_input]
    }
}
