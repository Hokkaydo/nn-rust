use facial_recognition::linalg::tensor_grad::Tensor;
#[cfg(test)]
#[test]
fn test_matmul() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let tensor_a = Tensor::new(data_a, &[2, 2]);
    let tensor_b = Tensor::new(data_b, &[2, 2]);
    let result = tensor_a.matmul(&tensor_b);
    let expected_data = vec![19.0, 22.0, 43.0, 50.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_mul() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let tensor_a = Tensor::new(data_a, &[2, 2]);
    let tensor_b = Tensor::new(data_b, &[2, 2]);
    let result = tensor_a * tensor_b;
    let expected_data = vec![5.0, 12.0, 21.0, 32.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}
