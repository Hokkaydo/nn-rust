use crate::linalg::autograd::grad_fn::TensorEWMulTensorFn;
use crate::linalg::tensor_grad::{Scalar, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub fn add_tt(a: &Tensor, b: &Tensor) -> Tensor {
    if a.is_scalar() {
        // TODO: handle grad
        return add_ts(b, a.storage.data[0]);
    }
    if b.is_scalar() {
        // TODO: handle grad
        return add_ts(a, b.storage.data[0]);
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

        Tensor::increment_indices(&mut indices, &a.shape());
    }

    let requires_grad = a.requires_grad || b.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(
                crate::linalg::autograd::grad_fn::TensorAddTensorFn {},
            ))
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new((*a).clone()), Rc::new((*b).clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}
pub fn add_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] + b);
    }
    let requires_grad = a.requires_grad;
    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for b addition not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(a.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}

pub fn sub_tt(a: &Tensor, b: &Tensor) -> Tensor {
    if a.is_scalar() {
        // TODO: handle grad
        return sub_st(a.storage.data[0], b);
    }
    if b.is_scalar() {
        // TODO: handle grad
        return sub_ts(a, b.storage.data[0]);
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

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            Some(Rc::new(
                crate::linalg::autograd::grad_fn::TensorSubTensorFn {},
            ))
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new((*a).clone()), Rc::new((*b).clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}
pub fn sub_ts(a: &Tensor, b: Scalar) -> Tensor {
    let result_data = a
        .storage
        .data
        .iter()
        .map(|x| x - b)
        .collect::<Vec<Scalar>>();
    let requires_grad = a.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for scalar subtraction not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(a.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}
pub fn sub_st(a: Scalar, b: &Tensor) -> Tensor {
    let result_data = b
        .storage
        .data
        .iter()
        .map(|x| a - x)
        .collect::<Vec<Scalar>>();
    let requires_grad = b.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: b.shape().to_vec(),
        strides: b.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for scalar subtraction not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(b.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}
pub fn mul_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] * b);
    }

    let requires_grad = a.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for scalar multiplication not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(a.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}

// element-wise multiplication
pub fn mul_tt_ews(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.shape, b.shape,
        "Shape mismatch for element-wise multiplication: {:?} vs {:?}",
        a.shape, b.shape
    );

    let total_elements: usize = a.shape.iter().product();
    let mut result_data = Vec::with_capacity(total_elements);

    // Use strides
    let mut indices = vec![0; a.shape.len()];
    for _ in 0..total_elements {
        let idx_a = a.compute_flat_index(&indices);
        let idx_b = b.compute_flat_index(&indices);

        result_data.push(a.storage.data[idx_a] * b.storage.data[idx_b]);

        Tensor::increment_indices(&mut indices, &a.shape);
    }
    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if a.requires_grad || b.requires_grad {
            Some(Rc::new(TensorEWMulTensorFn {
                lhs: Rc::new(a.clone()),
                rhs: Rc::new(b.clone()),
            }))
        } else {
            None
        },
        parents: if a.requires_grad || b.requires_grad {
            vec![Rc::new(a.clone()), Rc::new(b.clone())]
        } else {
            Vec::new()
        },
        requires_grad: a.requires_grad || b.requires_grad,
    }
}

pub fn div_ts(a: &Tensor, b: Scalar) -> Tensor {
    let mut result_data = Vec::with_capacity(a.shape().iter().product());
    for i in 0..a.storage.data.len() {
        result_data.push(a.storage.data[i] / b);
    }

    let requires_grad = a.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for scalar division not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(a.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}

pub fn div_st(a: Scalar, b: &Tensor) -> Tensor {
    let mut result_data = Vec::with_capacity(b.shape().iter().product());
    for i in 0..b.storage.data.len() {
        result_data.push(a / b.storage.data[i]);
    }

    let requires_grad = b.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: b.shape().to_vec(),
        strides: b.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for scalar division not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new(b.clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}

// element-wise multiplication
pub fn div_tt_ews(a: &Tensor, b: &Tensor) -> Tensor {
    if a.is_scalar() {
        // TODO: handle grad
        return div_st(a.storage.data[0], b);
    }
    if b.is_scalar() {
        // TODO: handle grad
        return div_ts(a, b.storage.data[0]);
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

        Tensor::increment_indices(&mut indices, &a.shape());
    }

    let requires_grad = a.requires_grad || b.requires_grad;

    Tensor {
        storage: Rc::new(Storage::new(result_data)),
        shape: a.shape().to_vec(),
        strides: a.strides.clone(),
        offset: 0,
        grad: RefCell::new(None),
        grad_fn: if requires_grad {
            None // Gradient function for division not implemented
        } else {
            None
        },
        parents: if requires_grad {
            vec![Rc::new((*a).clone()), Rc::new((*b).clone())]
        } else {
            Vec::new()
        },
        requires_grad,
    }
}
