use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_pow_grad() {
    let base = Tensor::with_grad(vec![2.0, 3.0, 4.0], &[3]);
    let exponent = 3.0;
    let pow = base.pow(exponent);
    pow.backward();
    let grad_base = base.grad().unwrap();
    let expected_grad_base = [12.0, 27.0, 48.0]; // Derivative of x^3 is 3*x^2
    let expected_shape = &[3];
    assert_eq!(grad_base.shape(), expected_shape);
    assert_eq!(grad_base.as_slice(), &expected_grad_base);
}
