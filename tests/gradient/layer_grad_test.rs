use nn_rs::helpers::metrics::mse;
use nn_rs::linalg::tensor::Tensor;
use nn_rs::nn::Layer;
use nn_rs::nn::linear::Linear;

#[cfg(test)]
#[test]
fn test_linear_layer_grad() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let linear = Linear::init(4, 2);
    let output = linear.forward(&input);
    let target = Tensor::new(vec![1.0, 1.0], &[1, 2]);
    let loss = mse(&target, &output);
    loss.backward();

    let [weights_grad, bias_grad] = &linear
        .parameters()
        .iter()
        .filter_map(|&x| x.grad())
        .collect::<Vec<_>>()[..]
    else {
        panic!("Should exist")
    };

    let expected_grad_weights_shape = &[4, 2];
    let expected_grad_bias_shape = &[1, 2];

    assert_eq!(weights_grad.shape(), expected_grad_weights_shape);
    assert_eq!(bias_grad.shape(), expected_grad_bias_shape);

    let expected_grad_weights = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let expected_grad_bias = vec![1.0, 1.0];

    assert_eq!(weights_grad.as_slice(), &expected_grad_weights);
    assert_eq!(bias_grad.as_slice(), &expected_grad_bias);
}
