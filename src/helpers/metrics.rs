use crate::linalg::tensor::{Scalar, Tensor};
/// Computes the negative log-likelihood loss, where the current tensor contains raw probabilities
/// # Arguments
/// * `target` - A tensor of the same shape as self, containing the target labels (one-hot encoded)
/// # Returns
/// A tensor containing the NLL loss value
pub fn nll_loss(target: &Tensor, pred: &Tensor) -> Tensor {
    let shape = &pred.shape;
    assert_eq!(shape.len(), 2, "Expected 2D tensor for input");
    assert_eq!(target.shape.len(), 2, "Expected 2D tensor for target");
    assert_eq!(
        shape[0], target.shape[0],
        "Input and target batch sizes must match"
    );
    assert_eq!(
        shape[1], target.shape[1],
        "Input and target must have the same number of classes"
    );
    let mut loss = 0.0;
    for (i, &value) in pred.storage.data.iter().enumerate() {
        let target_index = target.storage.data[i];
        if target_index == 1.0 {
            loss -= value.ln();
        }
    }
    // Normalize by batch size
    Tensor::new(vec![loss / (shape[0] as f32)], &[1])
}

pub fn mse(target: &Tensor, pred: &Tensor) -> Tensor {
    assert_eq!(
        target.shape(),
        pred.shape(),
        "Target and prediction tensors must have the same shape for MSE"
    );
    let sub = pred - target;
    println!("created sub tensor {}", sub.storage.id);
    let sq = sub.square();
    println!("created sq tensor {}", sq.storage.id);
    let s = sq.mean_scalar();
    println!("created mean tensor {}", s.storage.id);
    s
}

pub fn mae(target: &Tensor, pred: &Tensor) -> Tensor {
    assert_eq!(
        target.shape(),
        pred.shape(),
        "Target and prediction tensors must have the same shape for MAE"
    );
    (target - pred).abs().mean_scalar()
}

pub fn binary_cross_entropy(target: &Tensor, pred: &Tensor) -> Tensor {
    assert_eq!(
        target.shape(),
        pred.shape(),
        "Target and prediction tensors must have the same shape for binary cross-entropy"
    );
    let epsilon = 1e-12;
    let pred_clipped = pred.clamp(epsilon, 1.0 - epsilon);
    let out = -target * pred_clipped.log() - (1.0 - target) * (1.0 - pred_clipped).log();
    out.mean_scalar()
}

pub fn binary_cross_entropy_with_logits(target: &Tensor, pred: &Tensor) -> Tensor {
    assert_eq!(
        target.shape(),
        pred.shape(),
        "Target and prediction tensors must have the same shape for binary cross-entropy with logits"
    );

    let epsilon = 1e-12;
    let pred_clipped = pred.clamp(epsilon, 1.0 - epsilon);
    let out = target * &pred_clipped.sigmoid().log()
        + (1.0 - target) * (1.0 - &pred_clipped.sigmoid().log());
    -out.mean_scalar()
}

pub fn cross_entropy(target: &Tensor, pred: &Tensor) -> Tensor {
    nll_loss(target, &pred.log_softmax())
}

pub fn accuracy(target: &Tensor, pred: &Tensor) -> Scalar {
    assert_eq!(
        target.shape(),
        pred.shape(),
        "Target and prediction tensors must have the same shape for accuracy"
    );
    let mut correct = 0;
    for i in 0..target.shape()[0] {
        let target_class = target.storage.data[i] as usize;
        let predicted_class = pred.slice(0, i, 1).argmax_axis(0)[0];
        if target_class == predicted_class {
            correct += 1;
        }
    }
    correct as Scalar / target.shape()[0] as Scalar
}
