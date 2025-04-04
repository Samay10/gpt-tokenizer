# gpt-tokenizer

## Overview

This project implements two tokenization approaches: a `BasicTokenizer` and a more advanced `RegexBPETokenizer`. These tokenizers are designed to process text data by converting it into numerical tokens (encoding) and reconstructing the original text from those tokens (decoding). The tokenizers leverage Unicode encoding/decoding and Byte-Pair Encoding (BPE) techniques, with the latter incorporating regex-based splitting inspired by GPT-4's `cl100k_base` pattern.

The primary goal is to provide efficient and customizable tokenization for natural language processing tasks, with support for special tokens and verbose training options.

## Features

- **BasicTokenizer**: A simple tokenizer using Unicode byte-level encoding and BPE to merge frequent token pairs.
- **RegexBPETokenizer**: An advanced tokenizer that combines regex-based text splitting (using GPT-4's pattern) with BPE, supporting special tokens like `<|endoftext|>`.
- **Training**: Both tokenizers can be trained on custom text corpora to build vocabularies of a specified size.
- **Encoding/Decoding**: Convert text to token IDs and back, with handling for special tokens in `RegexBPETokenizer`.
- **Verbose Mode**: Optional logging of merge operations during training for debugging and insight.

## Requirements

- Python 3.x
- Libraries:
  - `re` (standard library) or `regex` (for advanced regex support)
  - `collections` (standard library)

Install the `regex` package if not already available:
```bash
pip install regex
```

## Usage

### BasicTokenizer
1. **Initialize**: Create an instance of BasicTokenizer.
2. **Train**: Train the tokenizer on a text corpus with a desired vocabulary size.
3. **Encode/Decode**: Use the trained tokenizer to encode text into token IDs and decode them back to text.

Example:

```python
tokenizer = BasicTokenizer()
with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer.train(text, vocab_size=100, verbose=True)
encoded = tokenizer.encode("Sample text here")
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
```

### RegexBPETokenizer
1. **Initialize**: Create a RegexTokenizer with an optional custom regex pattern, then pass it to RegexBPETokenizer with optional special tokens.
2. **Train**: Train on a text corpus, specifying vocabulary size.
3. **Encode/Decode**: Process text with support for special tokens.

Example:

```python
regex_tokenizer = RegexTokenizer()
tokenizer = RegexBPETokenizer(regex_tokenizer, special_tokens=["<|endoftext|>"])
with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer.train(text, vocab_size=100, verbose=True)
encoded = tokenizer.encode("Sample text <|endoftext|>", allowed_special="all")
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
```

## Code Structure

### BasicTokenizer:
- `__init__`: Initializes vocabulary and merge lists.
- `get_vocab_tokens`: Converts words to byte-level tokens.
- `get_stats`: Computes pair frequencies.
- `merge_vocab`: Merges frequent pairs.
- `train`: Builds the vocabulary through iterative merging.
- `encode`: Converts text to token IDs.
- `decode`: Reconstructs text from IDs.

### RegexBPETokenizer:
- `__init__`: Sets up regex tokenizer, vocab, and special tokens.
- `get_vocab_tokens`: Byte-level tokenization with end markers.
- `get_stats/merge_vocab`: Similar to BasicTokenizer but regex-aware.
- `train`: Incorporates regex splitting and special token handling.
- `encode`: Processes text with special token support.
- `decode`: Reconstructs text, preserving special tokens.
- `bpe/recover_merges`: Advanced BPE utilities for merge recovery.

## Notes

- The BasicTokenizer is simpler and faster but less flexible than RegexBPETokenizer.
- The RegexBPETokenizer uses GPT-4's cl100k_base regex pattern by default, improving tokenization of contractions (e.g., 's, 'll) and spacing.
- Special tokens in RegexBPETokenizer require explicit allowance during encoding to avoid errors.
- The code assumes UTF-8 encoded input files.

## Example File

The code includes an example using a file named taylorswift.txt. Replace this with your own text file for testing.

## Limitations

- Vocabulary size must be chosen carefully to balance efficiency and coverage.
- Error handling for malformed input is minimal; add checks as needed.
- Performance may degrade with very large corpora or vocab sizes.

## Contributing

Feel free to fork this repository, submit issues, or propose enhancements via pull requests. Suggestions for optimizing BPE merges or adding new features are welcome!

## The repository is secured under MIT License.
