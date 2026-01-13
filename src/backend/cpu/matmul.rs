use crate::backend::backend::MatMulOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl MatMulOps<Self> for CPUBackend {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[0], b_shape[0],
            "Incompatible shapes for matmul: {:?} and {:?}",
            a_shape, b_shape
        );
        let mut result_data = vec![0.0; 1];
        for i in 0..a_shape[0] {
            result_data[0] += a.as_slice()[i] * b.as_slice()[i];
        }
        Tensor::new(result_data, [1])
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[0], b_shape[0],
            "Incompatible shapes for matmul: {:?} and {:?}",
            a_shape, b_shape
        );
        let mut result_data = vec![0.0; b_shape[1]];
        for j in 0..b_shape[1] {
            for i in 0..a_shape[0] {
                result_data[j] += a.as_slice()[i] * b.as_slice()[i * b_shape[1] + j];
            }
        }
        Tensor::new(result_data, [b_shape[1]])
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[1], b_shape[0],
            "Incompatible shapes for matmul: {:?} and {:?}",
            a_shape, b_shape
        );
        let mut result_data = vec![0.0; a_shape[0]];
        for i in 0..a_shape[0] {
            for j in 0..a_shape[1] {
                result_data[i] += a.as_slice()[i * a_shape[1] + j] * b.as_slice()[j];
            }
        }
        Tensor::new(result_data, [a_shape[0]])
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[1], b_shape[0],
            "Incompatible shapes for matmul: {:?} and {:?}",
            a_shape, b_shape
        );
        let mut result_data = vec![0.0; a_shape[0] * b_shape[1]];
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                for k in 0..a_shape[1] {
                    result_data[i * b_shape[1] + j] +=
                        a.as_slice()[i * a_shape[1] + k] * b.as_slice()[k * b_shape[1] + j];
                }
            }
        }
        Tensor::new(result_data, [a_shape[0], b_shape[1]])
    }
}
