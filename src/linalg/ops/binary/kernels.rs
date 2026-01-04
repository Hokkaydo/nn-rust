use crate::linalg::autograd::grad_fn::binary::{AddGradFn, DivGradFn, EWSMultGradFn, SubGradFn};
use crate::linalg::tensor_grad::{InternalTensor, Scalar, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub fn add_tt(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = None;
    if a.is_scalar() {
        out = Some(add_ts(b, a.storage.data[0]));
    }
    if b.is_scalar() {
        out = Some(add_ts(a, b.storage.data[0]));
    }
    if let Some(mut out) = out {
        if a.requires_grad || b.requires_grad {
            out.set_grad_metadata(Rc::new(AddGradFn), vec![a.clone(), b.clone()])
        }
        return out;
    }

    assert_eq!(
        a.shape(),
        b.shape(),
        "Shape mismatch: {:?} vs {:?}. Broadcasting not yet implemented.",
        a.shape(),
        b.shape()
    );

    let total_elements: usize = a.shape().iter().product();
    let mut result_data = Vec::with_capacity(total_elements);

    let mut indices = vec![0; a.shape().len()];
    for _ in 0..total_elements {
        let idx_a = a.compute_flat_index(&indices);
        let idx_b = b.compute_flat_index(&indices);

        result_data.push(a.storage.data[idx_a] + b.storage.data[idx_b]);

        Tensor::increment_indices(&mut indices, a.shape());
    }

    let requires_grad = a.requires_grad || b.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(AddGradFn))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone(), b.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}

pub fn add_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] + b);
    }
    let requires_grad = a.requires_grad;
    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(AddGradFn))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}
pub fn sub_tt(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = None;
    if a.is_scalar() {
        out = Some(sub_st(a.storage.data[0], b));
    }
    if b.is_scalar() {
        out = Some(sub_ts(a, b.storage.data[0]));
    }

    if let Some(mut out) = out {
        if a.requires_grad || b.requires_grad {
            out.set_grad_metadata(Rc::new(SubGradFn(true, true)), vec![a.clone(), b.clone()])
        }
        return out;
    }

    assert_eq!(
        Tensor::reduce_shape(a.shape()),
        Tensor::reduce_shape(b.shape()),
        "Shapes must match for subtraction ({:?} != {:?})",
        a.shape(),
        b.shape()
    );
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] - b.storage.data[i]);
    }

    let requires_grad = a.requires_grad || b.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(SubGradFn(true, true)))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone(), b.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}
pub fn sub_ts(a: &Tensor, b: Scalar) -> Tensor {
    let result_data = a
        .storage
        .data
        .iter()
        .map(|x| x - b)
        .collect::<Vec<Scalar>>();
    let requires_grad = a.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(SubGradFn(true, false)))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}
pub fn sub_st(a: Scalar, b: &Tensor) -> Tensor {
    let result_data = b
        .storage
        .data
        .iter()
        .map(|x| a - x)
        .collect::<Vec<Scalar>>();
    let requires_grad = b.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: b.shape().to_vec(),
        strides: b.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(SubGradFn(false, true)))
        } else {
            None
        },
        parents: if requires_grad {
            vec![b.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}

pub fn mul_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] * b);
    }

    let requires_grad = a.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(EWSMultGradFn(Some(b))))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}

// element-wise multiplication
pub fn mul_tt_ews(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = None;

    if a.is_scalar() {
        out = Some(mul_ts(b, a.storage.data[0]));
    }
    if b.is_scalar() {
        out = Some(mul_ts(a, b.storage.data[0]));
    }
    if let Some(mut out) = out {
        if a.requires_grad || b.requires_grad {
            out.set_grad_metadata(Rc::new(EWSMultGradFn(None)), vec![a.clone(), b.clone()])
        };
        return out;
    }

    assert_eq!(
        Tensor::reduce_shape(a.shape()),
        Tensor::reduce_shape(b.shape()),
        "Shape mismatch for element-wise multiplication: {:?} vs {:?}",
        a.shape,
        b.shape
    );

    let total_elements: usize = a.shape().iter().product();
    let mut result_data = Vec::with_capacity(total_elements);

    // Use strides
    let mut a_indices = vec![0; a.shape().len()];
    let mut b_indices = vec![0; b.shape().len()];
    for _ in 0..total_elements {
        let idx_a = a.compute_flat_index(&a_indices);
        let idx_b = b.compute_flat_index(&b_indices);

        result_data.push(a.storage.data[idx_a] * b.storage.data[idx_b]);

        Tensor::increment_indices(&mut a_indices, a.shape());
        Tensor::increment_indices(&mut b_indices, b.shape());
    }
    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().into(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if a.requires_grad || b.requires_grad {
            Some(Rc::new(EWSMultGradFn(None)))
        } else {
            None
        },
        parents: if a.requires_grad || b.requires_grad {
            vec![a.clone(), b.clone()]
        } else {
            Vec::new()
        },
        requires_grad: a.requires_grad || b.requires_grad,
    }
    .into()
}

pub fn div_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] / b);
    }

    let requires_grad = a.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(DivGradFn(None, Some(b))))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}

pub fn div_st(a: Scalar, b: &Tensor) -> Tensor {
    let mut result_data = Vec::with_capacity(b.shape().iter().product());
    for i in 0..b.storage.data.len() {
        result_data.push(a / b.storage.data[i]);
    }

    let requires_grad = b.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: b.shape().to_vec(),
        strides: b.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(DivGradFn(Some(a), None)))
        } else {
            None
        },
        parents: if requires_grad {
            vec![b.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}

// element-wise division
pub fn div_tt_ews(a: &Tensor, b: &Tensor) -> Tensor {
    if a.is_scalar() {
        let mut out = div_st(a.storage.data[0], b);
        if a.requires_grad || b.requires_grad {
            out.set_grad_metadata(
                Rc::new(DivGradFn(Some(a.as_scalar()), None)),
                vec![a.clone(), b.clone()],
            )
        };
        return out;
    }

    if b.is_scalar() {
        let mut out = div_ts(a, b.storage.data[0]);
        if a.requires_grad || b.requires_grad {
            out.set_grad_metadata(
                Rc::new(DivGradFn(None, Some(b.as_scalar()))),
                vec![a.clone(), b.clone()],
            )
        };
        return out;
    }

    assert_eq!(
        a.shape(),
        b.shape(),
        "Shape mismatch: {:?} vs {:?}. Broadcasting not yet implemented.",
        a.shape(),
        b.shape()
    );

    let total_elements: usize = a.shape().iter().product();
    let mut result_data = Vec::with_capacity(total_elements);

    let mut indices = vec![0; a.shape().len()];
    for _ in 0..total_elements {
        let idx_a = a.compute_flat_index(&indices);
        let idx_b = b.compute_flat_index(&indices);

        result_data.push(a.storage.data[idx_a] / b.storage.data[idx_b]);

        Tensor::increment_indices(&mut indices, a.shape());
    }

    let requires_grad = a.requires_grad || b.requires_grad;

    InternalTensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(DivGradFn(None, None)))
        } else {
            None
        },
        parents: if requires_grad {
            vec![a.clone(), b.clone()]
        } else {
            Vec::new()
        },
        requires_grad,
    }
    .into()
}
