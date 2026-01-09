pub(crate) mod activation;
pub(crate) mod binary;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod shape;
pub(crate) mod unary;

use crate::linalg::tensor::Tensor;

pub(crate) trait GradFn {
    /// Applies the gradient function to the given gradient output tensor_old and returns the gradients for each parent tensor_old.
    fn apply(&self, grad_output: &Tensor) -> Vec<Tensor>;
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub(crate) struct NotImplementedGradFn(pub(crate) &'static str);

impl GradFn for NotImplementedGradFn {
    fn apply(&self, _grad_output: &Tensor) -> Vec<Tensor> {
        panic!("{}'s gradient is not defined", self.0);
    }
}
#[macro_export]
macro_rules! not_implemented_grad_fn {
    ($name:expr) => {
        Some(Rc::new(
            $crate::linalg::autograd::grad_fn::NotImplementedGradFn($name),
        ))
    };
}
