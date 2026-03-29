//! Example: two-layer MLP regression on synthetic data (CPU + autograd + SGD).
//!
//! Target: `y ≈ 2*x0 - x1 + noise`. Loss decreases over epochs if the run succeeds.

use std::rc::Rc;

use rusty_ai_autograd::{backward, Variable};
use rusty_ai_core::Tensor;
use rusty_ai_ml::Sgd;
use rusty_ai_nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 42u32;
    let l1 = Linear::new(2, 8, &mut seed)?;
    let l2 = Linear::new(8, 1, &mut seed)?;

    // All trainable leaves: same Rc nodes each epoch so gradients apply to stored weights.
    let params: Vec<Rc<Variable>> = vec![
        Rc::clone(&l1.weight),
        Rc::clone(&l1.bias),
        Rc::clone(&l2.weight),
        Rc::clone(&l2.bias),
    ];

    let opt = Sgd::new(0.15);

    // Synthetic dataset: y ≈ 2*x0 - x1 + small noise
    let n = 200usize;
    let mut xs = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    let mut s = 7u32;
    for _ in 0..n {
        let x0 = (s.wrapping_mul(1103515245).wrapping_add(12345) % 1000) as f32 / 1000.0;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let x1 = (s % 1000) as f32 / 1000.0;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        xs.push(x0);
        xs.push(x1);
        ys.push(x0 * 2.0 - x1 + 0.05 * ((s % 17) as f32 / 17.0 - 0.5));
    }

    let x_train = Tensor::from_vec(xs, vec![n, 2])?;
    let y_train = Tensor::from_vec(ys, vec![n, 1])?;

    for epoch in 0..80 {
        let x = Variable::leaf(x_train.clone());
        let t = y_train.clone();

        let h = Variable::relu(&l1.forward(&x)?);
        let pred = l2.forward(&h)?;

        let loss = Variable::mse(&pred, &t)?;
        for p in &params {
            p.zero_grad();
        }
        backward(&loss)?;
        opt.step(&params)?;

        if epoch % 20 == 0 || epoch == 79 {
            println!("epoch {epoch:3}  loss = {:.6}", loss.data().data()[0]);
        }
    }

    Ok(())
}
