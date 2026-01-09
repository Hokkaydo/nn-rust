use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_broadcast_add_grad() {
    let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let b = Tensor::with_grad(vec![7.0, 8.0, 9.0], &[3, 1]);

    let c = a.broadcast_add(&b);

    let loss = c.sum();
    loss.backward();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    let expected_grad_a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let expected_grad_b = [2.0, 2.0, 2.0];
    let expected_shape_a = &[3, 2];
    let expected_shape_b = &[3, 1];

    assert_eq!(grad_a.shape(), expected_shape_a);
    assert_eq!(grad_b.shape(), expected_shape_b);
    assert_eq!(grad_a.as_slice(), &expected_grad_a);
    assert_eq!(grad_b.as_slice(), &expected_grad_b);
}

#[cfg(test)]
#[test]
fn test_sub_grad() {
    let a = Tensor::with_grad(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let b = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = &a - &b;

    let loss = c.sum();
    loss.backward();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    let expected_grad_a = [1.0, 1.0, 1.0, 1.0];
    let expected_grad_b = [-1.0, -1.0, -1.0, -1.0];
    let expected_shape = &[2, 2];

    assert_eq!(grad_a.shape(), expected_shape);
    assert_eq!(grad_b.shape(), expected_shape);
    assert_eq!(grad_a.as_slice(), &expected_grad_a);
    assert_eq!(grad_b.as_slice(), &expected_grad_b);
}
