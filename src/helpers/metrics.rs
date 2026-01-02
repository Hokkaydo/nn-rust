use crate::linalg::tensor_old::Tensor;

pub fn mse(target: &Tensor, pred: &Tensor) -> f32 {
    assert_eq!(
        target.shape, pred.shape,
        "Target and prediction tensors must have the same shape for MSE"
    );
    (target - pred).square().mean()
}

pub fn mae(target: &Tensor, pred: &Tensor) -> f32 {
    assert_eq!(
        target.shape, pred.shape,
        "Target and prediction tensors must have the same shape for MAE"
    );
    (target - pred).abs().mean()
}

pub fn binary_cross_entropy(target: &Tensor, pred: &Tensor) -> f32 {
    assert_eq!(
        target.shape, pred.shape,
        "Target and prediction tensors must have the same shape for binary cross-entropy"
    );
    let epsilon = 1e-12;
    let pred_clipped = pred.map(|x| x.clamp(epsilon, 1.0 - epsilon));
    let out = -target * pred_clipped.log() - (1.0 - target) * (1.0 - pred_clipped).log();
    out.mean()
}

pub fn binary_cross_entropy_with_logits(target: &Tensor, pred: &Tensor) -> f32 {
    assert_eq!(
        target.shape, pred.shape,
        "Target and prediction tensors must have the same shape for binary cross-entropy with logits"
    );
    fn sigmoid(tensor: &Tensor) -> Tensor {
        tensor.map(|x| 1.0 / (1.0 + (-x).exp()))
    }
    let epsilon = 1e-12;
    let pred_clipped = pred.map(|x| x.clamp(epsilon, 1.0 - epsilon));
    let out = target.element_wise_multiply(&sigmoid(&pred_clipped).log())
        + (1.0 - target).element_wise_multiply(&(1.0 - sigmoid(&pred_clipped)).log());
    -out.mean()
}

pub fn cross_entropy(target: &Tensor, pred: &Tensor) -> f32 {
    pred.log_softmax().nll_loss(target)
}
pub fn accuracy(target: &Tensor, pred: &Tensor) -> f32 {
    assert_eq!(
        target.shape, pred.shape,
        "Target and prediction tensors must have the same shape for accuracy"
    );
    let mut correct = 0;
    for i in 0..target.shape[0] {
        let target_class = target.data[i] as usize;
        let predicted_class = pred.get_level(i).argmax();
        if target_class == predicted_class {
            correct += 1;
        }
    }
    correct as f32 / target.shape()[0] as f32
}
