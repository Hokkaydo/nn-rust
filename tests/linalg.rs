use nn_rs::linalg::tensor_old::Tensor;

#[cfg(test)]
#[test]
fn tensor_new_set_get() {
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let mut tensor = Tensor::new(data, vec![2, 2]);
    assert_eq!(tensor.get(&[0, 0]), 1.0);
    tensor.set(&[0, 0], 5.0);
    assert_eq!(tensor.get(&[0, 0]), 5.0);

    assert_eq!(tensor[&[0, 1]], 2.0);
    tensor.set(&[0, 1], 3.0);
    assert_eq!(tensor[&[0, 1]], 3.0);

    tensor[&[1, 0]] = 6.0;
    assert_eq!(tensor[&[1, 0]], 6.0);
}

#[cfg(test)]
#[test]
fn tensor_add() {
    let data1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];

    let tensor1 = Tensor::new(data1, vec![2, 5]);
    let tensor2 = Tensor::new(data2, vec![2, 5]);
    let results = vec![
        &tensor1 + &tensor2,
        &tensor1 + tensor2.clone(),
        tensor1.clone() + &tensor2,
        tensor1 + tensor2,
    ];
    let expected_data = vec![5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0];

    for result in results {
        assert_eq!(result.shape, vec![2, 5]);
        for i in 0..2 {
            for j in 0..5 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 5 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_sub() {
    let data1 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let data2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let tensor1 = Tensor::new(data1, vec![2, 5]);
    let tensor2 = Tensor::new(data2, vec![2, 5]);
    let results = vec![
        &tensor1 - &tensor2,
        &tensor1 - tensor2.clone(),
        tensor1.clone() - &tensor2,
        tensor1 - tensor2,
    ];
    let expected_data = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];

    for result in results {
        assert_eq!(result.shape, vec![2, 5]);
        for i in 0..2 {
            for j in 0..5 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 5 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_sub_scalar() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let scalar = 5.0;

    let results = [
        [&tensor - scalar, tensor.clone() - scalar],
        [scalar - &tensor, scalar - tensor],
    ];
    let expected_data = [[-4.0, -3.0, -2.0, -1.0], [4.0, 3.0, 2.0, 1.0]];

    println!("{:?}", results);
    for k in 0..2 {
        assert_eq!(results[0][k].shape, vec![2, 2]);
        assert_eq!(results[1][k].shape, vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(results[0][k].get(&[i, j]), expected_data[0][i * 2 + j]);
                assert_eq!(results[1][k].get(&[i, j]), expected_data[1][i * 2 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_neg() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let results = vec![-&tensor, -tensor.clone()];
    let expected_data = vec![-1.0, -2.0, -3.0, -4.0];

    for result in results {
        assert_eq!(result.shape, vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_sub_mismatched_shape() {
    let data1 = vec![5.0, 6.0, 7.0, 8.0];
    let data2 = vec![1.0, 2.0, 3.0];

    let tensor1 = Tensor::new(data1, vec![2, 2]);
    let tensor2 = Tensor::new(data2, vec![3]);

    let _result = tensor1 - tensor2;
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_add_mismatched_shape() {
    let data1 = vec![0.0, 1.0, 2.0, 3.0];
    let data2 = vec![4.0, 5.0, 6.0];

    let tensor1 = Tensor::new(data1, vec![2, 2]);
    let tensor2 = Tensor::new(data2, vec![3]);

    let _result = tensor1 + tensor2;
}

#[cfg(test)]
#[test]
fn tensor_scalar_add() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let scalar = 5.0;

    let results = vec![
        &tensor + scalar,
        tensor.clone() + scalar,
        scalar + &tensor,
        scalar + tensor,
    ];
    let expected_data = vec![6.0, 7.0, 8.0, 9.0];
    for result in results {
        assert_eq!(result.shape, vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_mul() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];

    let tensor1 = Tensor::new(data1, vec![2, 2]);
    let tensor2 = Tensor::new(data2, vec![2, 2]);
    let results = vec![
        &tensor1 * &tensor2,
        &tensor1 * tensor2.clone(),
        tensor1.clone() * &tensor2,
        tensor1 * tensor2,
    ];
    let expected_data = vec![19.0, 22.0, 43.0, 50.0];
    for result in results {
        assert_eq!(result.shape, vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_mul_mismatched_shape() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let tensor1 = Tensor::new(data1, vec![2, 2]);
    let tensor2 = Tensor::new(data2, vec![3, 2]);

    let _result = tensor1 * tensor2;
}

#[cfg(test)]
#[test]
fn tensor_mul_vector() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data1, vec![2, 2]);
    let data2 = vec![5.0, 6.0];
    let vec = Tensor::new(data2, vec![2]);

    let result = tensor * vec;
    let expected_data = vec![17.0, 39.0];

    for i in 0..2 {
        assert_eq!(result[&[i]], expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn vector_mul_tensor() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data1, vec![2, 2]);
    let data2 = vec![5.0, 6.0];
    let vec = Tensor::new(data2, vec![2]);

    let result = vec * tensor;
    let expected_data = vec![23.0, 34.0];

    for i in 0..2 {
        assert_eq!(result[&[i]], expected_data[i]);
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_vec_mul_mismatched_shape() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data1, vec![2, 2]);
    let data2 = vec![4.0, 5.0, 6.0];
    let vec = Tensor::new(data2, vec![3]);

    let _result = tensor * vec;
}

#[cfg(test)]
#[test]
#[should_panic]
fn vec_tensor_mul_mismatched_shape() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data1, vec![2, 2]);
    let data2 = vec![4.0, 5.0, 6.0];
    let vec = Tensor::new(data2, vec![3]);

    let _result = vec * tensor;
}

#[cfg(test)]
#[test]
fn element_wise_mul() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let vec1 = Tensor::new(data1, vec![4]);
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    let vec2 = Tensor::new(data2, vec![4]);
    let result = vec1 * vec2;

    let expected_data = vec![5.0, 12.0, 21.0, 32.0];
    for i in 0..4 {
        assert_eq!(result.get(&[i]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn element_wise_mul_mismatched_shape() {
    let data1 = vec![1.0, 2.0, 3.0];
    let vec1 = Tensor::new(data1, vec![3]);
    let data2 = vec![4.0, 5.0];
    let vec2 = Tensor::new(data2, vec![2]);

    let _result = vec1 * vec2;
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_mul_shape_too_big() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor1 = Tensor::new(data1, vec![2, 2, 2]);
    let data2 = vec![5.0, 6.0, 7.0];
    let tensor2 = Tensor::new(data2, vec![3]);

    let _result = tensor1 * tensor2;
}

#[cfg(test)]
#[test]
fn tensor_scalar_mul() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let scalar = 2.0;

    let results = vec![
        &tensor * scalar,
        tensor.clone() * scalar,
        scalar * &tensor,
        scalar * tensor,
    ];
    let expected_data = vec![2.0, 4.0, 6.0, 8.0];
    for result in results {
        assert_eq!(result.shape, vec![2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_div_by_scalar() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let scalar = 2.0;

    let result = tensor / scalar;
    let expected_data = vec![0.5, 1.0, 1.5, 2.0];

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_div_by_zero() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let _result = tensor / 0.0;
}

#[cfg(test)]
#[test]
fn tensor_squeeze() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let result = tensor.squeeze();
    assert_eq!(result.shape, vec![2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), tensor.get(&[i, j]));
        }
    }

    let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor2 = Tensor::new(data2, vec![3, 2, 1]);
    let result2 = tensor2.squeeze();
    assert_eq!(result2.shape, vec![3, 2]);

    let data3 = vec![1.0];
    let mat3 = Tensor::new(data3, vec![1]);
    let result3 = mat3.squeeze();
    assert_eq!(result3.shape, vec![1]);
}

#[cfg(test)]
#[test]
fn tensor_sum_dim() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![3, 2]);

    let result = tensor.sum_axis(0);
    assert_eq!(result.shape, vec![1, 2]);
    assert_eq!(result.get(&[0]), 9.0);
    assert_eq!(result.get(&[1]), 12.0);
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_init_mismatched_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let _tensor = Tensor::new(data, vec![3, 2]);
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_access_mismatched_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let _val = tensor.get(&[2, 0, 1]);
}

#[cfg(test)]
#[test]
fn tensor_get_level() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = Tensor::new(data, vec![3, 2, 2]);

    let level0 = tensor.get_level(1);
    assert_eq!(level0.shape, vec![1, 2, 2]);
    let expected_data = vec![5.0, 6.0, 7.0, 8.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(level0.get(&[0, i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_get_levels() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let tensor = Tensor::new(data, vec![4, 2, 2]);

    let sub_tensor = tensor.get_levels(&[1, 3]);
    assert_eq!(sub_tensor.shape, vec![2, 2, 2]);
    let expected_data = vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(sub_tensor.get(&[0, i, j]), expected_data[i * 2 + j]);
            assert_eq!(sub_tensor.get(&[1, i, j]), expected_data[i * 2 + j + 4]);
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_get_level_out_of_bounds() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![3, 2, 1]);

    let _level0 = tensor.get_level(3);
}

#[cfg(test)]
#[test]
fn tensor_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![4, 1]);

    let mean = tensor.mean();
    assert_eq!(mean, 2.5);
}

#[cfg(test)]
#[test]
fn tensor_max() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let max = tensor.max();
    assert_eq!(max, 4.0);
}

#[cfg(test)]
#[test]
fn tensor_square() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

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
fn tensor_argmax() {
    let data = vec![1.0, 2.0, 5.0, 4.0];
    let tensor = Tensor::new(data, vec![4, 1]);

    let argmax = tensor.argmax();
    assert_eq!(argmax, 2);
}

#[cfg(test)]
#[test]
fn tensor_log() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data.clone(), vec![2, 2]);

    let result = tensor.log();
    let expected_data = data.iter().map(|&x| x.ln()).collect::<Vec<_>>();

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let result = tensor.sum();
    assert_eq!(result, 10.0);
}

#[cfg(test)]
#[test]
fn tensor_abs() {
    let data = vec![-1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

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
fn tensor_map() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let result = tensor.map(|x| x * 2.0);
    let expected_data = vec![2.0, 4.0, 6.0, 8.0];

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn tensor_reduce() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let result = tensor.reduce(0.0, |acc, x| acc + x);
    assert_eq!(result, 10.0);
}

#[cfg(test)]
#[test]
fn tensor_reduce_default() {
    let data = vec![];
    let tensor = Tensor::new(data, vec![0]);

    let result = tensor.reduce(0.0, |acc, x| acc + x);
    assert_eq!(result, 0.0);
}

#[cfg(test)]
#[test]
fn tensor_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);

    let result = tensor.transpose();
    let expected_data = vec![1.0, 3.0, 2.0, 4.0];

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_transpose_out_of_dims() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor = Tensor::new(data, vec![2, 2, 2]);

    let _result = tensor.transpose();
}

#[cfg(test)]
#[test]
fn tensor_broadcast_add() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![2, 3]);
    let data2 = vec![5.0, 6.0];
    let vec1 = Tensor::new(data2, vec![2, 1]);

    let result = [tensor.broadcast_add(&vec1), vec1.broadcast_add(&tensor)];
    let expected_data = vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0];

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(result[0].get(&[i, j]), expected_data[i * 3 + j]);
            assert_eq!(result[1].get(&[i, j]), expected_data[i * 3 + j]);
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_broadcast_add_shape_mismatch() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let data2 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0];
    let vec1 = Tensor::new(data2, vec![3, 3]);

    let _result = tensor.broadcast_add(&vec1);
}

#[cfg(test)]
#[test]
fn tensor_element_wise_mul() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    let tensor2 = Tensor::new(data2, vec![2, 2]);

    let result = tensor.element_wise_multiply(&tensor2);
    let expected_data = vec![5.0, 12.0, 21.0, 32.0];

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get(&[i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor_element_wise_mul_shape_mismatch() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2]);
    let data2 = vec![5.0, 6.0, 7.0];
    let tensor2 = Tensor::new(data2, vec![3]);

    let _result = tensor.element_wise_multiply(&tensor2);
}
