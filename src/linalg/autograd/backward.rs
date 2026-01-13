use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor::{InternalTensor, Scalar, Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

impl InternalTensor {
    pub(crate) fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
}

impl Tensor {
    pub(crate) fn set_grad_metadata(
        &mut self,
        grad_fn: Rc<dyn GradFn>,
        parents: Vec<Weak<InternalTensor>>,
    ) {
        let inner = Rc::make_mut(&mut self.0);
        inner.grad_fn = Some(grad_fn);
        inner.parents = parents;
    }
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

    pub fn detach(&mut self) {
        let inner = Rc::make_mut(&mut self.0);
        inner.grad = RefCell::new(None);
        inner.grad_fn = None;
        inner.parents = Vec::new();
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Performs backpropagation to compute gradients for all tensors in the computation graph
    /// that have `requires_grad` set to true.
    pub fn backward(&self) {
        assert!(self.requires_grad);

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

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

            let parents = t
                .parents
                .iter()
                .filter_map(|x| x.upgrade())
                .filter(|p| p.requires_grad);

            println!(
                "Len parents: {}, Len parent_grads: {}",
                parents.clone().count(),
                parent_grads.len()
            );

            for (parent, g) in parents.zip(parent_grads) {
                if !parent.requires_grad {
                    continue;
                }

                let g = if g.shape() != parent.shape() {
                    g.sum_to_shape(parent.shape())
                } else {
                    g
                };

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

    /// Sums the tensor to match the specified shape by summing over dimensions where the target shape has size 1.
    /// # Arguments
    /// * `shape` - The target shape to sum to.
    /// # Returns
    /// A new tensor summed to the specified shape.
    pub fn sum_to_shape(&self, shape: &[usize]) -> Tensor {
        let mut result = self.clone();
        for (i, &dim) in Tensor::reduce_shape(self.shape()).iter().enumerate() {
            if shape[i] == 1 && dim != 1 {
                result = result.sum_axis(i);
            }
        }
        result.reshape(shape)
    }

    /// Expands the tensor by repeating its data along a new dimension.
    /// # Arguments
    /// * `axis` - The axis along which to expand the tensor.
    /// * `size` - The size of the new dimension.
    /// # Returns
    /// A new tensor with the expanded dimension.
    pub fn expand_dim(&self, axis: usize, size: usize) -> Tensor {
        assert!(axis <= self.shape.len(), "expand_dim: invalid axis");

        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, size);

        let old_shape = &self.shape;
        let old_strides = Tensor::compute_strides(old_shape);
        let new_strides = Tensor::compute_strides(&new_shape);

        let mut new_data = vec![0.0; new_shape.iter().product()];

        for (idx, data_slot) in new_data.iter_mut().enumerate() {
            // Convert flat index → multi-index in new tensor
            let mut rem = idx;
            let mut new_indices = vec![0; new_shape.len()];
            for i in 0..new_shape.len() {
                new_indices[i] = rem / new_strides[i];
                rem %= new_strides[i];
            }

            // Remove expanded axis to get old tensor index
            let mut old_indices = new_indices;
            old_indices.remove(axis);

            let old_flat = old_indices
                .iter()
                .zip(old_strides.iter())
                .map(|(i, s)| i * s)
                .sum::<usize>();

            *data_slot = self.storage.data[old_flat];
        }

        Tensor::new(new_data, &new_shape)
    }
}

fn build_topo(t: &Tensor, visited: &mut HashSet<TensorId>, out: &mut Vec<Tensor>) {
    let id = t.storage.id;
    if visited.contains(&id) {
        return;
    }
    visited.insert(id);

    println!(
        "Visiting tensor id: {}, requires_grad: {}",
        id, t.requires_grad
    );
    for p in &t.parents {
        if let Some(p) = p.upgrade() {
            println!("parent requires_grad: {}", p.requires_grad);
            build_topo(&Tensor(p), visited, out);
        } else {
            println!("parent has been dropped");
        }
    }

    out.push(t.clone());
}
