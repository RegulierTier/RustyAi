//! Mini-GPT next-token training on a tiny byte sequence (cross-entropy + Adam, CPU autograd).
//!
//! TODO: load real corpus from disk / chunk long files for practical code-LM experiments.
//!
//! Uses [`TrainableMiniGpt`] with the same forward as [`MiniGpt`](rusty_ai_llm::MiniGpt); loss should
//! decrease over a few steps on this small repeated pattern.

use std::rc::Rc;

use rusty_ai_autograd::{backward, Variable};
use rusty_ai_llm::{MiniGpt, MiniGptConfig, TrainableMiniGpt};
use rusty_ai_ml::Adam;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 7u32;
    let cfg = MiniGptConfig {
        vocab_size: 64,
        d_model: 32,
        n_heads: 4,
        n_layers: 1,
        ffn_dim: 64,
        max_seq: 32,
    };
    let m = MiniGpt::random(cfg, &mut seed)?;
    let model = TrainableMiniGpt::from_mini_gpt(&m)?;
    let params: Vec<Rc<Variable>> = model.parameters();

    let mut opt = Adam::new(0.01);

    let bytes: Vec<u8> = b"hello mini gpt".to_vec();
    let token_ids: Vec<usize> = bytes.iter().map(|&b| b as usize).collect();

    for step in 0..30 {
        for p in &params {
            p.zero_grad();
        }
        let logits = model.forward(&token_ids)?;
        let loss = Variable::cross_entropy_next_token(&logits, &token_ids)?;
        backward(&loss)?;
        opt.step(&params)?;

        if step % 5 == 0 || step == 29 {
            println!("step {step:3}  loss = {:.6}", loss.data().data()[0]);
        }
    }

    Ok(())
}
