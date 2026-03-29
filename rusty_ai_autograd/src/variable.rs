//! [`Variable`]: a node in the computation graph with optional gradients.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use crate::grad_utils::{
    gelu_backward, gelu_forward, layer_norm_backward, layer_norm_forward,
    softmax_last_dim_backward, softmax_last_forward, sum_grad_to_shape,
};
use crate::structural::{merge_heads_tensor, split_heads_tensor};
use rusty_ai_core::{
    add, matmul, mse, mul, relu, sub, sum_axis_0, transpose_2d, transpose_batched_last2, Tensor,
    TensorError,
};

/// A tensor value participating in autograd, plus the operation that produced it (if any).
///
/// - **Data** live in a [`RefCell`] so optimizers can replace weights after `backward`.
/// - **Gradients** are stored when `backward` runs; call [`Variable::zero_grad`] before each
///   new forward pass if you reuse the same parameter nodes.
pub struct Variable {
    storage: RefCell<Tensor>,
    grad: RefCell<Option<Tensor>>,
    op: Op,
}

/// Discriminated backward rule: each variant knows how to route `grad_output` to parents.
#[derive(Clone)]
enum Op {
    Leaf,
    Add(Rc<Variable>, Rc<Variable>),
    BiasAdd(Rc<Variable>, Rc<Variable>),
    MatMul(Rc<Variable>, Rc<Variable>),
    Mul(Rc<Variable>, Rc<Variable>),
    Relu(Rc<Variable>),
    Gelu(Rc<Variable>),
    LayerNorm(Rc<Variable>, f32),
    SoftmaxLastDim(Rc<Variable>),
    Reshape(Rc<Variable>, Vec<usize>),
    TransposeBatchedLast2(Rc<Variable>),
    /// Upper-future positions in the last two axes (causal attention scores) masked with `-1e9`.
    CausalMask(Rc<Variable>),
    /// Embedding table `(vocab, d)`; `indices` has length `seq` (one row picked per position).
    EmbeddingGather(Rc<Variable>, Vec<usize>),
    /// `(batch, seq, d_model) -> (batch*heads, seq, d_head)`.
    SplitHeads(Rc<Variable>, usize, usize, usize),
    /// `(batch*heads, seq, d_head) -> (batch, seq, d_model)`.
    MergeHeads(Rc<Variable>, usize, usize, usize),
    /// Mean cross-entropy for next-token prediction: logits `(1, seq, vocab)`, `targets.len() == seq`.
    CrossEntropyNextToken(Rc<Variable>, Vec<usize>),
    Mse(Rc<Variable>, Tensor),
}

impl Variable {
    pub fn leaf(data: Tensor) -> Rc<Self> {
        Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Leaf,
        })
    }

    pub fn data(&self) -> Ref<'_, Tensor> {
        self.storage.borrow()
    }

    pub fn data_mut(&self) -> RefMut<'_, Tensor> {
        self.storage.borrow_mut()
    }

    pub fn set_data(&self, t: Tensor) {
        *self.storage.borrow_mut() = t;
    }

    pub fn add(a: &Rc<Variable>, b: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = add(&a.data(), &b.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Add(Rc::clone(a), Rc::clone(b)),
        }))
    }

    pub fn bias_add(x: &Rc<Variable>, bias: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = add(&x.data(), &bias.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::BiasAdd(Rc::clone(x), Rc::clone(bias)),
        }))
    }

    pub fn matmul(a: &Rc<Variable>, b: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = matmul(&a.data(), &b.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::MatMul(Rc::clone(a), Rc::clone(b)),
        }))
    }

    /// Elementwise multiplication with broadcasting (same rules as `rusty_ai_core::mul`).
    pub fn mul(a: &Rc<Variable>, b: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = mul(&a.data(), &b.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Mul(Rc::clone(a), Rc::clone(b)),
        }))
    }

    pub fn relu(v: &Rc<Variable>) -> Rc<Self> {
        let data = relu(&v.data());
        if !crate::grad_enabled() {
            return leaf_detached(data);
        }
        Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Relu(Rc::clone(v)),
        })
    }

    /// GELU activation (tanh approximation, matches `rusty_ai_nn::gelu`).
    pub fn gelu(v: &Rc<Variable>) -> Rc<Self> {
        let data = gelu_forward(&v.data());
        if !crate::grad_enabled() {
            return leaf_detached(data);
        }
        Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Gelu(Rc::clone(v)),
        })
    }

    /// Layer normalization over the last axis (no γ/β).
    pub fn layer_norm(v: &Rc<Variable>, eps: f32) -> Result<Rc<Self>, TensorError> {
        let data = layer_norm_forward(&v.data(), eps)?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::LayerNorm(Rc::clone(v), eps),
        }))
    }

    /// Softmax over the last dimension.
    pub fn softmax_last_dim(v: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = softmax_last_forward(&v.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::SoftmaxLastDim(Rc::clone(v)),
        }))
    }

    /// Same layout, new shape (total element count unchanged).
    pub fn reshape(v: &Rc<Variable>, new_shape: &[usize]) -> Result<Rc<Self>, TensorError> {
        let old_shape = v.data().shape().to_vec();
        let data = v.data().reshape(new_shape)?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Reshape(Rc::clone(v), old_shape),
        }))
    }

    /// Swap last two dimensions on each batch slice `(B, M, N) -> (B, N, M)`.
    pub fn transpose_batched_last2(v: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = transpose_batched_last2(&v.data())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::TransposeBatchedLast2(Rc::clone(v)),
        }))
    }

    /// Gather rows from `table` shaped `(vocab, d)`; output `(1, seq, d)` for `indices.len() == seq`.
    pub fn embedding_gather(
        table: &Rc<Variable>,
        indices: &[usize],
    ) -> Result<Rc<Self>, TensorError> {
        let td = table.data();
        let shape = td.shape();
        if shape.len() != 2 {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: shape.to_vec(),
                    to: vec![],
                },
            ));
        }
        let v = shape[0];
        let d = shape[1];
        let seq = indices.len();
        let mut out = vec![0.0f32; seq * d];
        let w = td.data();
        for (t, &id) in indices.iter().enumerate() {
            let row = id.min(v - 1);
            for j in 0..d {
                out[t * d + j] = w[row * d + j];
            }
        }
        let data = Tensor::from_vec(out, vec![1, seq, d])?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::EmbeddingGather(Rc::clone(table), indices.to_vec()),
        }))
    }

    pub fn split_heads(
        x: &Rc<Variable>,
        batch: usize,
        heads: usize,
        d_head: usize,
    ) -> Result<Rc<Self>, TensorError> {
        let data = split_heads_tensor(&x.data(), heads, d_head)?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::SplitHeads(Rc::clone(x), batch, heads, d_head),
        }))
    }

    pub fn merge_heads(
        x: &Rc<Variable>,
        batch: usize,
        heads: usize,
    ) -> Result<Rc<Self>, TensorError> {
        let d_head = x.data().shape()[2];
        let data = merge_heads_tensor(&x.data(), batch, heads)?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::MergeHeads(Rc::clone(x), batch, heads, d_head),
        }))
    }

    /// Causal mask on attention score tensor `(batch_heads, seq, seq)`: future columns set to `-1e9`.
    pub fn causal_mask_scores(scores: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let sd = scores.data();
        let s = sd.shape();
        if s.len() != 3 || s[1] != s[2] {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: s.to_vec(),
                    to: vec![0, 0, 0],
                },
            ));
        }
        let bh = s[0];
        let seq = s[2];
        let mut data = sd.data().to_vec();
        let stride = seq * seq;
        for b in 0..bh {
            for i in 0..seq {
                for j in 0..seq {
                    if j > i {
                        data[b * stride + i * seq + j] = -1e9f32;
                    }
                }
            }
        }
        let out = Tensor::from_vec(data, s.to_vec())?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(out));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(out),
            grad: RefCell::new(None),
            op: Op::CausalMask(Rc::clone(scores)),
        }))
    }

    /// Next-token language-model loss: for each position `t < seq-1`, predicts `targets[t+1]` from `logits[0,t,:]`.
    /// Scalar mean over `seq-1` positions.
    pub fn cross_entropy_next_token(
        logits: &Rc<Variable>,
        targets: &[usize],
    ) -> Result<Rc<Self>, TensorError> {
        let logd = logits.data();
        let s = logd.shape();
        if s.len() != 3 || s[0] != 1 {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: s.to_vec(),
                    to: vec![1, 0, 0],
                },
            ));
        }
        let seq = s[1];
        let vocab = s[2];
        if targets.len() != seq {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![targets.len()],
                    to: vec![seq],
                },
            ));
        }
        let n_pred = seq.saturating_sub(1);
        if n_pred == 0 {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![seq],
                    to: vec![2],
                },
            ));
        }
        let ld = logd.data();
        let mut loss = 0.0f32;
        for t in 0..seq - 1 {
            let y = targets[t + 1].min(vocab - 1);
            let base = t * vocab;
            let row = &ld[base..base + vocab];
            let mut max = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for &v in row.iter() {
                sum_exp += (v - max).exp();
            }
            let log_p_y = row[y] - max - sum_exp.ln();
            loss += -log_p_y;
        }
        loss /= n_pred as f32;
        let data = Tensor::scalar(loss);
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::CrossEntropyNextToken(Rc::clone(logits), targets.to_vec()),
        }))
    }

    pub fn mse(pred: &Rc<Variable>, target: &Tensor) -> Result<Rc<Self>, TensorError> {
        let data = mse(&pred.data(), target)?;
        if !crate::grad_enabled() {
            return Ok(leaf_detached(data));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Mse(Rc::clone(pred), target.clone()),
        }))
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().clone()
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }
}

fn leaf_detached(data: Tensor) -> Rc<Variable> {
    Rc::new(Variable {
        storage: RefCell::new(data),
        grad: RefCell::new(None),
        op: Op::Leaf,
    })
}

fn acc_grad(v: &Variable, g: &Tensor) -> Result<(), TensorError> {
    let mut cell = v.grad.borrow_mut();
    match &mut *cell {
        Some(acc) => {
            *acc = add(acc, g)?;
        }
        None => {
            *cell = Some(g.clone());
        }
    }
    Ok(())
}

pub fn backward(loss: &Rc<Variable>) -> Result<(), TensorError> {
    let g = Tensor::scalar(1.0);
    backward_grad(loss, &g)
}

fn backward_grad(v: &Rc<Variable>, grad: &Tensor) -> Result<(), TensorError> {
    acc_grad(v, grad)?;

    match &v.op {
        Op::Leaf => Ok(()),
        Op::Add(a, b) => {
            backward_grad(a, grad)?;
            backward_grad(b, grad)?;
            Ok(())
        }
        Op::BiasAdd(x, b) => {
            backward_grad(x, grad)?;
            let gb = sum_axis_0(grad)?;
            backward_grad(b, &gb)?;
            Ok(())
        }
        Op::MatMul(a, b) => {
            let ra = a.data().shape().len();
            let rb = b.data().shape().len();
            if ra == 2 && rb == 2 {
                let ga = matmul(grad, &transpose_2d(&b.data())?)?;
                let gb = matmul(&transpose_2d(&a.data())?, grad)?;
                backward_grad(a, &ga)?;
                backward_grad(b, &gb)?;
            } else if ra == 3 && rb == 3 {
                let bt = transpose_batched_last2(&b.data())?;
                let ga = matmul(grad, &bt)?;
                let at = transpose_batched_last2(&a.data())?;
                let gb = matmul(&at, grad)?;
                backward_grad(a, &ga)?;
                backward_grad(b, &gb)?;
            } else {
                return Err(TensorError::Shape(
                    rusty_ai_core::ShapeError::MatmulIncompatible {
                        left: a.data().shape().to_vec(),
                        right: b.data().shape().to_vec(),
                    },
                ));
            }
            Ok(())
        }
        Op::Mul(a, b) => {
            let ga = mul(grad, &b.data())?;
            let gb = mul(grad, &a.data())?;
            let ga = sum_grad_to_shape(&ga, a.data().shape())?;
            let gb = sum_grad_to_shape(&gb, b.data().shape())?;
            backward_grad(a, &ga)?;
            backward_grad(b, &gb)?;
            Ok(())
        }
        Op::Relu(x) => {
            let mask = relu_mask(&x.data());
            let gx = mul(grad, &mask)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::Gelu(x) => {
            let gx = gelu_backward(grad, &x.data())?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::LayerNorm(x, eps) => {
            let gx = layer_norm_backward(grad, &x.data(), *eps)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::SoftmaxLastDim(x) => {
            let y = softmax_last_forward(&x.data())?;
            let gx = softmax_last_dim_backward(grad, &y)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::Reshape(a, old_shape) => {
            let ga = grad.reshape(old_shape)?;
            backward_grad(a, &ga)?;
            Ok(())
        }
        Op::TransposeBatchedLast2(a) => {
            let ga = transpose_batched_last2(grad)?;
            backward_grad(a, &ga)?;
            Ok(())
        }
        Op::CausalMask(pre) => {
            let pd = pre.data();
            let s = pd.shape();
            if s.len() != 3 {
                return Err(TensorError::Shape(
                    rusty_ai_core::ShapeError::InvalidReshape {
                        from: s.to_vec(),
                        to: vec![],
                    },
                ));
            }
            let bh = s[0];
            let seq = s[2];
            let mut g = grad.data().to_vec();
            let stride = seq * seq;
            for b in 0..bh {
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i {
                            g[b * stride + i * seq + j] = 0.0;
                        }
                    }
                }
            }
            let gt = Tensor::from_vec(g, s.to_vec())?;
            backward_grad(pre, &gt)?;
            Ok(())
        }
        Op::EmbeddingGather(table, indices) => {
            let td = table.data();
            let shape = td.shape();
            let v = shape[0];
            let d = shape[1];
            let gd = grad.data();
            let mut acc = vec![0.0f32; v * d];
            for (t, &id) in indices.iter().enumerate() {
                let row = id.min(v - 1);
                for j in 0..d {
                    acc[row * d + j] += gd[t * d + j];
                }
            }
            let gt = Tensor::from_vec(acc, shape.to_vec())?;
            backward_grad(table, &gt)?;
            Ok(())
        }
        Op::SplitHeads(x, batch, heads, _d_head) => {
            let gx = merge_heads_tensor(grad, *batch, *heads)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::MergeHeads(x, _batch, heads, d_head) => {
            let gx = split_heads_tensor(grad, *heads, *d_head)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::CrossEntropyNextToken(logits, targets) => {
            let ld = logits.data();
            let s = ld.shape();
            let seq = s[1];
            let vocab = s[2];
            let n_pred = (seq - 1) as f32;
            let ldata = ld.data();
            let g0 = grad.data().first().copied().unwrap_or(1.0);
            let mut g = vec![0.0f32; seq * vocab];
            for t in 0..seq - 1 {
                let y = targets[t + 1].min(vocab - 1);
                let base = t * vocab;
                let row = &ldata[base..base + vocab];
                let mut max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max {
                        max = v;
                    }
                }
                let mut sum_exp = 0.0f32;
                let mut p = vec![0.0f32; vocab];
                for (i, &v) in row.iter().enumerate() {
                    let e = (v - max).exp();
                    p[i] = e;
                    sum_exp += e;
                }
                let inv = 1.0 / sum_exp.max(1e-12);
                for pi in p.iter_mut() {
                    *pi *= inv;
                }
                let scale = g0 / n_pred;
                for (i, &pi) in p.iter().enumerate() {
                    let delta = if i == y { 1.0 } else { 0.0 };
                    g[base + i] += scale * (pi - delta);
                }
            }
            let gt = Tensor::from_vec(g, s.to_vec())?;
            backward_grad(logits, &gt)?;
            Ok(())
        }
        Op::Mse(pred, target) => {
            let n = pred.data().numel().max(1) as f32;
            let diff = sub(&pred.data(), target)?;
            let g0 = grad.data().first().copied().unwrap_or(1.0);
            let scale = mul(&diff, &Tensor::scalar(2.0 / n * g0))?;
            backward_grad(pred, &scale)?;
            Ok(())
        }
    }
}

fn relu_mask(x: &Tensor) -> Tensor {
    let mut out = x.data().to_vec();
    for v in &mut out {
        if *v > 0.0 {
            *v = 1.0;
        } else {
            *v = 0.0;
        }
    }
    Tensor::from_vec(out, x.shape().to_vec()).expect("same shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_grad<F>(f: F, x: &[f32], eps: f32) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let mut g = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            g[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
        }
        g
    }

    #[test]
    fn mlp_backward_runs() {
        let x = Variable::leaf(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let w = Variable::leaf(Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]).unwrap());
        let h = Variable::matmul(&x, &w).unwrap();
        let y = Variable::relu(&h);
        let target = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let loss = Variable::mse(&y, &target).unwrap();
        backward(&loss).unwrap();
        assert!(w.grad().is_some());
        assert!(x.grad().is_some());
    }

    #[test]
    fn gelu_grad_matches_numeric() {
        let x0 = vec![-0.5f32, 0.0, 0.7, 1.2];
        let t = Tensor::from_vec(x0.clone(), vec![4]).unwrap();
        let v = Variable::leaf(t.clone());
        let y = Variable::gelu(&v);
        let target = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        let loss = Variable::mse(&y, &target).unwrap();
        backward(&loss).unwrap();
        let ag = v.grad().unwrap();
        let ng = approx_grad(
            |xv| {
                let tt = Tensor::from_vec(xv.to_vec(), vec![4]).unwrap();
                mse(&gelu_forward(&tt), &target).unwrap().data()[0]
            },
            &x0,
            1e-3,
        );
        for (i, (a, n)) in ag.data().iter().zip(ng.iter()).enumerate() {
            assert!((*a - *n).abs() < 5e-2, "i={i} ag={a} ng={n}");
        }
    }

    #[test]
    fn mul_broadcast_backward() {
        let a = Variable::leaf(Tensor::from_vec(vec![2.0f32], vec![1, 1]).unwrap());
        let b = Variable::leaf(Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], vec![1, 3]).unwrap());
        let p = Variable::mul(&a, &b).unwrap();
        let target = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let loss = Variable::mse(&p, &target).unwrap();
        backward(&loss).unwrap();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn cross_entropy_next_token_backward() {
        let vocab = 8usize;
        let seq = 5usize;
        let logits = Variable::leaf(
            Tensor::from_vec(
                (0..(seq * vocab)).map(|i| (i as f32) * 0.1 - 1.0).collect(),
                vec![1, seq, vocab],
            )
            .unwrap(),
        );
        let targets = vec![0usize, 1, 2, 3, 4];
        let loss = Variable::cross_entropy_next_token(&logits, &targets).unwrap();
        backward(&loss).unwrap();
        assert!(logits.grad().is_some());
    }

    #[test]
    fn embedding_gather_backward() {
        let table = Variable::leaf(
            Tensor::from_vec((0..20).map(|i| i as f32 * 0.1).collect(), vec![5, 4]).unwrap(),
        );
        let indices = vec![1usize, 2, 1];
        let e = Variable::embedding_gather(&table, &indices).unwrap();
        let target = Tensor::zeros(&[1, 3, 4], rusty_ai_core::DType::F32).unwrap();
        let loss = Variable::mse(&e, &target).unwrap();
        backward(&loss).unwrap();
        let g = table.grad().unwrap();
        assert!(g.data().iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn split_merge_grad_runs() {
        let x = Variable::leaf(Tensor::from_vec(vec![1.0f32; 2 * 3 * 8], vec![2, 3, 8]).unwrap());
        let s = Variable::split_heads(&x, 2, 2, 4).unwrap();
        let m = Variable::merge_heads(&s, 2, 2).unwrap();
        let target = Tensor::zeros(&[2, 3, 8], rusty_ai_core::DType::F32).unwrap();
        let loss = Variable::mse(&m, &target).unwrap();
        backward(&loss).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn matmul_3d_backward_runs() {
        let a = Variable::leaf(
            Tensor::from_vec((0..12).map(|i| i as f32 * 0.1).collect(), vec![2, 2, 3]).unwrap(),
        );
        let b = Variable::leaf(
            Tensor::from_vec((0..24).map(|i| i as f32 * 0.05).collect(), vec![2, 3, 4]).unwrap(),
        );
        let c = Variable::matmul(&a, &b).unwrap();
        let target = Tensor::zeros(&[2, 2, 4], rusty_ai_core::DType::F32).unwrap();
        let loss = Variable::mse(&c, &target).unwrap();
        backward(&loss).unwrap();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }
}
