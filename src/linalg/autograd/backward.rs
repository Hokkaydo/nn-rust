use crate::linalg::tensor_grad::{InternalTensor, Scalar, Tensor};
use std::collections::HashSet;
use std::rc::Rc;

impl InternalTensor {
    pub(crate) fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
}

impl Tensor {
    /// Creates a new tensor with the given data and shape, with `requires_grad` set to true.
    /// # Arguments
    /// * `data` - A vector containing the tensor data.
    /// * `shape` - A slice representing the shape of the tensor.
    /// # Returns
    /// A new tensor with `requires_grad` set to true.
    pub fn with_grad(data: Vec<Scalar>, shape: &[usize]) -> Self {
        InternalTensor::new(data, shape)
            .with_requires_grad(true)
            .into()
    }

    /// Returns the gradient if it exists as an `Option`.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().as_ref().cloned()
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Performs backpropagation to compute gradients for all tensors in the computation graph
    /// that have `requires_grad` set to true.
    /// Gradients are accumulated in the `grad` field of each tensor.
    pub fn backward(self: &Self) {
        self.backward_with_options(true);
    }

    /// Performs backpropagation to compute gradients for all tensors in the computation graph
    /// that have `requires_grad` set to true.
    /// # Arguments
    /// * `retain_graph` If true, retains the computation graph after backward pass,
    /// else clears gradients of intermediate tensors.
    pub fn backward_with_options(self: &Self, retain_graph: bool) {
        assert!(self.requires_grad);

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        if !retain_graph {
            for t in &topo {
                t.zero_grad();
            }
        }

        *self.grad.borrow_mut() = Some(Tensor::ones(&self.shape));

        for t in topo.into_iter().rev() {
            let grad_out = match &*t.grad.borrow() {
                Some(g) => g.clone(),
                None => continue,
            };

            let grad_fn = match &t.grad_fn {
                Some(f) => f,
                None => continue,
            };

            let parent_grads = grad_fn.apply(&grad_out);

            for (parent, g) in t.parents.iter().zip(parent_grads) {
                if !parent.requires_grad {
                    continue;
                }

                let mut parent_grad = parent.grad.borrow_mut();
                match &mut *parent_grad {
                    Some(existing) => {
                        *existing = &*existing + &g;
                    }
                    None => {
                        *parent_grad = Some(g);
                    }
                }
            }
        }
    }
}

fn build_topo(t: &Tensor, visited: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
    let id = Rc::as_ptr(&t.0) as usize;
    if visited.contains(&id) {
        return;
    }
    visited.insert(id);

    for p in &t.parents {
        build_topo(p, visited, out);
    }

    out.push(t.clone());
}
