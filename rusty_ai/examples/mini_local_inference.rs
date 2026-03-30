//! Lädt das eingecheckte Mini‑Checkpoint aus Speicher (kein Netz, keine externen Pfade zur Laufzeit) und generiert Text.

use rusty_ai_llm::{generate_from_ids, load_minigpt_checkpoint_bytes, ByteTokenizer};

const CONFIG_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/mini_local/config.json"
));
const MODEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/mini_local/model.safetensors"
));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_minigpt_checkpoint_bytes(CONFIG_JSON, MODEL_BYTES)?;
    let prompt = "hello ";
    let ids = ByteTokenizer::encode(prompt);
    let mut seed = 12345u32;
    let out = generate_from_ids(&model, &ids, 48, 0.9, 0.95, &mut seed)?;
    println!("{}", ByteTokenizer::decode(&out));
    Ok(())
}
