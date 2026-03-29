//! Fixed 256-token vocabulary: raw bytes as `usize` token ids (UTF-8 safe at byte level).
//!
//! FIXME: poor token efficiency vs BPE for source code — prefer `gpt2-bpe` or a code tokenizer for serious use.

/// Byte-level tokenizer: one token per byte (`0..256`).
///
/// Decoding uses [`String::from_utf8_lossy`] so invalid UTF-8 sequences become the replacement character.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// Vocabulary size (all possible byte values).
    pub const VOCAB_SIZE: usize = 256;

    /// UTF-8 string → sequence of byte values as token indices.
    pub fn encode(s: &str) -> Vec<usize> {
        s.bytes().map(|b| b as usize).collect()
    }

    /// Token indices → lossy UTF-8 string (bytes clamped to `0..=255`).
    pub fn decode(ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&x| x.min(255) as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
