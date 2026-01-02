use facial_recognition::linalg::tensor_grad::Tensor;

#[cfg(test)]
#[test]
fn test_neg() {
    let data = vec![1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::new(data, &[2, 2]);
    let result = -tensor;
    let expected_data = vec![-1.0, 2.0, -3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_pow() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, &[2, 2]);
    let exponent = 3.0;
    let result = tensor.pow(exponent);
    let expected_data = vec![1.0, 8.0, 27.0, 64.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_square() {
    let data = vec![1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::new(data, &[2, 2]);
    let result = tensor.square();
    let expected_data = vec![1.0, 4.0, 9.0, 16.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_abs() {
    let data = vec![-1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::new(data, &[2, 2]);
    let result = tensor.abs();
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_clamp() {
    let data = vec![-1.0, 0.5, 2.0, 3.5];
    let tensor = Tensor::new(data, &[2, 2]);
    let min = 0.0;
    let max = 2.0;
    let result = tensor.clamp(min, max);
    let expected_data = vec![0.0, 0.5, 2.0, 2.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_log() {
    let data = vec![
        1.0,
        std::f32::consts::E,
        std::f32::consts::E.powf(2.0),
        std::f32::consts::E.powf(3.0),
    ];
    let tensor = Tensor::new(data, &[2, 2]);
    let result = tensor.log();
    let expected_data = vec![0.0, 1.0, 2.0, 3.0];
    for i in 0..2 {
        for j in 0..2 {
            assert!((result.get(&[i, j]) - expected_data[i * 2 + j]).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[test]
fn test_argmax() {
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let tensor = Tensor::new(data, &[5]);
    let result = tensor.argmax_axis(0);
    let expected_index = 3; // Index of the maximum value (5.0)
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_index);
}

#[cfg(test)]
#[test]
fn test_argmax2d() {
    let data_2d = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let tensor_2d = Tensor::new(data_2d, &[2, 3]);
    let result_2d = tensor_2d.argmax_axis(1);
    let expected_indices_2d = vec![1, 2]; // Indices of max values in each row
    assert_eq!(result_2d.len(), 2);
    for i in 0..2 {
        assert_eq!(result_2d[i], expected_indices_2d[i]);
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_argmax_invalid_axis() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::new(data, &[3]);
    let _result = tensor.argmax_axis(2); // Invalid axis for 1D tensor
}

#[cfg(test)]
#[test]
fn test_sqrt() {
    let data = vec![1.0, 4.0, 9.0, 16.0];
    let tensor = Tensor::new(data, &[2, 2]);
    let result = tensor.sqrt();
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert!((result.get(&[i, j]) - expected_data[i * 2 + j]).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[test]
fn test_exp() {
    let data = vec![0.0, 1.0, 2.0];
    let tensor = Tensor::new(data, &[3]);
    let result = tensor.exp();
    let expected_data = vec![1.0, std::f32::consts::E, std::f32::consts::E.powf(2.0)];
    for i in 0..3 {
        assert!((result.get(&[i]) - expected_data[i]).abs() < 1e-6);
    }
}
