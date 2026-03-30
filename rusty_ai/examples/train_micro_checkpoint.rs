//! Trainiert ein [`MiniGptConfig::micro_local`]‑Modell auf einem kleinen Byte‑Korpus und schreibt
//! `assets/mini_local/config.json` + `model.safetensors` (Maintainer: nach Änderungen erneut ausführen und committen).
//!
//! [`backward`](rusty_ai_autograd::backward) nutzt einen expliziten Heap-Stack (kein rekursiver
//! Aufruf-Stack), sodass das Training auf dem Main-Thread auch unter Windows stabil läuft.

use std::path::PathBuf;
use std::rc::Rc;

use rusty_ai_autograd::{backward, Variable};
use rusty_ai_llm::{
    load_minigpt_checkpoint_bytes, save_minigpt_checkpoint, MiniGpt, MiniGptConfig,
    TrainableMiniGpt,
};
use rusty_ai_ml::Adam;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("mini_local");

    let cfg = MiniGptConfig::micro_local();
    let mut seed = 42u32;
    let m = MiniGpt::random(cfg, &mut seed)?;
    let model = TrainableMiniGpt::from_mini_gpt(&m)?;
    let params: Vec<Rc<Variable>> = model.parameters();

    let mut opt = Adam::new(0.008);

    let corpus: &[u8] = b"hello mini transformer local byte lm repeat ";
    let token_ids: Vec<usize> = corpus.iter().map(|&b| b as usize).collect();

    const STEPS: usize = 40;
    for step in 0..STEPS {
        for p in &params {
            p.zero_grad();
        }
        let logits = model.forward(&token_ids)?;
        let loss = Variable::cross_entropy_next_token(&logits, &token_ids)?;
        backward(&loss)?;
        opt.step(&params)?;

        if step % 5 == 0 || step + 1 == STEPS {
            println!("step {step:3}  loss = {:.6}", loss.data().data()[0]);
        }
    }

    let trained = model.to_mini_gpt();
    save_minigpt_checkpoint(&out_dir, &trained)?;

    let json = std::fs::read_to_string(out_dir.join("config.json"))?;
    let weights = std::fs::read(out_dir.join("model.safetensors"))?;
    let loaded = load_minigpt_checkpoint_bytes(&json, &weights)?;
    assert_eq!(loaded.cfg.vocab_size, cfg.vocab_size);

    println!("Wrote checkpoint under {}", out_dir.display());
    Ok(())
}
