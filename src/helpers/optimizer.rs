use crate::linalg::tensor::Tensor;

pub fn gradient_descent(learning_rate: f32) -> Box<dyn Fn(&Tensor, &Tensor) -> Tensor> {
    Box::new(move |val: &Tensor, grad: &Tensor| val - &(grad * learning_rate))
}
