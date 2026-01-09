use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_mean_scalar_grad() {
    let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mean = a.mean_scalar();
    mean.backward();
    let grad_a = a.grad().unwrap();
    let expected_grad_a = [0.25, 0.25, 0.25, 0.25];
    let expected_shape = &[2, 2];
    assert_eq!(Tensor::reduce_shape(grad_a.shape()), expected_shape);
    assert_eq!(grad_a.as_slice(), &expected_grad_a);
}
