use crate::backend::backend::Backend;
use crate::linalg::tensor::Tensor;

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Computes the mean of the tensor along specified axes
    /// # Arguments
    /// * `axes` - A slice of axes along which to compute the mean
    /// # Returns
    /// A new tensor containing the mean values
    pub fn mean(&self, axes: &[usize]) -> Self {
        assert!(axes.len() <= NDIM, "Too many axes for mean");
        for &axis in axes {
            assert!(axis < NDIM, "Axis out of bounds for mean");
        }
        B::mean(self, Some(axes))
    }

    /// Computes the mean of all elements in the tensor
    /// # Returns
    /// A new tensor containing the mean value
    pub fn mean_scalar(&self) -> Self {
        B::mean(self, None)
    }

    /// Computes the L2 norm of the tensor
    /// # Returns
    /// A new tensor containing the L2 norm
    pub fn norm(&self) -> Self {
        self.square().sum().sqrt()
    }

    /// Slices the tensor along the specified axis
    /// and returns a new tensor containing the slice.
    /// # Arguments
    /// * `axis` - The axis along which to slice the tensor.
    /// * `start` - The starting index of the slice (inclusive).
    /// * `len` - The length of the slice.
    /// # Returns
    /// A new tensor containing the slice.
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> Self {
        assert!(axis < NDIM, "Axis out of bounds");
        assert!(start + len <= self.shape()[axis], "Invalid slice indices");
        B::slice(self, axis, start, len)
    }

    /// Gathers elements from the tensor along specified axis using provided indices
    /// # Arguments
    /// * `axis` - The axis along which to gather elements
    /// * `indices` - A vector containing the indices to gather
    /// # Returns
    /// A new tensor containing the gathered elements
    pub fn gather(&self, axis: usize, indices: &[usize]) -> Self {
        assert!(axis < NDIM);

        let axis_dim = self.shape()[axis];
        for &idx in indices {
            assert!(idx < axis_dim, "Gather index out of bounds");
        }

        B::gather(self, axis, indices)
    }

    /// Computes the indices of the maximum value in the tensor
    /// # Arguments
    /// * `axis` - The axis along which to compute the argmax
    /// # Returns
    /// A vector containing the indices of the maximum values along the specified axis
    /// # Example
    /// ```rust
    /// use nn_rs::linalg::tensor::Tensor;
    /// let tensor = Tensor::new(vec![4.0, 3.0, 6.0, 1.0, 5.0, 2.0], &vec![2, 3]);
    /// // 4 3 6
    /// // 1 5 2
    /// assert_eq!(tensor.argmax(0), vec![0, 1, 0]); // Max values in each column are 4, 5, 6
    /// assert_eq!(tensor.argmax(1), vec![2, 1]);    // Max values in each row are 6, 5
    /// ```
    pub fn argmax(&self, axis: usize) -> Tensor<B, { NDIM - 1 }> {
        assert!(axis <= NDIM, "Too many axes for argmax");
        B::argmax(self, axis)
    }

    /// Computes the maximum value in the tensor along specified axes
    /// # Arguments
    /// * `axes` - A slice of axes along which to compute the maximum
    /// # Returns
    /// A new tensor containing the maximum values
    pub fn max(&self, axes: &[usize]) -> Self {
        assert!(axes.len() <= NDIM, "Too many axes for max");
        for &axis in axes {
            assert!(axis < NDIM, "Axis out of bounds for max");
        }
        B::max(self, Some(axes))
    }

    /// Computes the minimum value in the tensor along specified axes
    /// # Arguments
    /// * `axes` - A slice of axes along which to compute the minimum
    /// # Returns
    /// A new tensor containing the minimum values
    pub fn min(&self, axes: &[usize]) -> Self {
        assert!(axes.len() <= NDIM, "Too many axes for min");
        for &axis in axes {
            assert!(axis < NDIM, "Axis out of bounds for min");
        }
        B::min(self, Some(axes))
    }

    /// Computes the maximum value in the tensor
    /// # Returns
    /// The maximum value
    pub fn max_scalar(&self) -> Self {
        B::max(self, None)
    }

    /// Computes the minimum value in the tensor
    /// # Returns
    /// The minimum value
    pub fn min_scalar(&self) -> Self {
        B::min(self, None)
    }

    /// Computes the sum of all elements in the tensor
    /// # Returns
    /// A tensor containing the sum of all elements
    pub fn sum(&self) -> Self {
        B::sum(self, None)
    }

    pub fn sum_axis(&self, axes: &[usize]) -> Self {
        assert!(axes.len() < NDIM, "Too many axes for sum");
        for &axis in axes {
            assert!(axis < NDIM, "Axis out of bounds for sum");
        }
        B::sum(self, Some(axes))
    }
}
