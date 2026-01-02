use crate::linalg::tensor_grad::{Scalar, Tensor};
use std::rc::Rc;

pub(crate) trait GradFn {
    /// Applies the gradient function to the given gradient output tensor_old and returns the gradients for each parent tensor_old.
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

pub(crate) struct TensorAddTensorFn;
pub(crate) struct TensorSubTensorFn;
pub(crate) struct TensorNegTensorFn;
pub(crate) struct TensorTransposeFn;
pub(crate) struct TensorPowFn {
    pub exponent: Scalar,
}
pub(crate) struct TensorMatMulTensorFn {
    pub lhs: Rc<Tensor>,
    pub rhs: Rc<Tensor>,
}
// Element-wise multiplication gradient function
pub(crate) struct TensorEWMulTensorFn {
    pub lhs: Rc<Tensor>,
    pub rhs: Rc<Tensor>,
}

impl GradFn for TensorAddTensorFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

impl GradFn for TensorSubTensorFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), -grad_output.clone()]
    }
}
impl GradFn for TensorNegTensorFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![-grad_output.clone()]
    }
}

impl GradFn for TensorTransposeFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let new_shape = grad_output.shape.clone();
        vec![grad_output.transpose()]
    }
}

impl GradFn for TensorPowFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let exponent = self.exponent;
        let grad_input = grad_output * &(grad_output.pow(exponent - 1.0) * exponent);
        vec![grad_input]
    }
}

impl GradFn for TensorMatMulTensorFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = grad_output.matmul(&self.rhs.transpose());
        let grad_b = self.lhs.transpose().matmul(grad_output);

        vec![grad_a, grad_b]
    }
}

impl GradFn for TensorEWMulTensorFn {
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let grad_a = grad_output * &*self.rhs;
        let grad_b = grad_output * &*self.lhs;

        vec![grad_a, grad_b]
    }
}
