use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_matmul_grad() {
    // Create two tensors with requires_grad = true
    let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::with_grad(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    // Perform matrix multiplication
    let c = a.matmul(&b);

    // Assume some loss function and compute gradient
    let loss = c.sum();
    loss.backward();

    // Check gradients
    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    // Expected gradients
    let expected_grad_a = [11.0, 15.0, 11.0, 15.0];
    let expected_grad_b = [4.0, 4.0, 6.0, 6.0];
    let expected_shape = &[2, 2];

    assert_eq!(grad_a.shape(), expected_shape);
    assert_eq!(grad_b.shape(), expected_shape);
    assert_eq!(grad_a.as_slice(), &expected_grad_a);
    assert_eq!(grad_b.as_slice(), &expected_grad_b);
}
