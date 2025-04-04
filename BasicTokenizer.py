# Importing the libraries
# re - regex for defining text formats for easy parsing
# collections: defaultdict: stores the initialized variables in a dictionary & Counter: count the frequency of words
import re
from collections import defaultdict, Counter

# BasicTokenizer: this class creates a tokenizer using unicode encoding/decoding + Byte-Pair Encoding. Once this is done
# we train the model for encoding and decoding a piece of text.
class BasicTokenizer:
  def __init__(self):
    self.vocab = {}
    self.inverse_vocab = {}
    self.bpe_merges = []

  # encode the characters and get the tokens for the vocabulary
  def get_vocab_tokens(self, word):
    byte_seq = list(word.encode("utf-8"))
    return [f"<{b}>" for b in byte_seq] + ["</w>"]

  # store the pairs and their frequencies
  def get_stats(self, tokens):
    pairs = defaultdict(int)
    for token, freq in tokens.items():
      symbols = token.split()
      for i in range(len(symbols) - 1):
        pairs[(symbols[i], symbols[i+1])] += freq

    return pairs

  # merge the frequently occuring pairs and replace them with un-used tokens
  def merge_vocab(self, pair, tokens):
    new_tokens = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for token in tokens:
      new_token = pattern.sub(''.join(pair), token)
      new_tokens[new_token] = tokens[token]

    return new_tokens

  # train the model 
  def train(self, text, vocab_size, verbose=False):
    words = text.strip().split()
    word_freq = Counter(words)
    tokens = {" ".join(self.get_vocab_tokens(word)): freq for word, freq in word_freq.items()}

    while len(self.vocab) < vocab_size:
      pairs = self.get_stats(tokens)
      if not pairs:
        break

      best = max(pairs, key=pairs.get)
      tokens = self.merge_vocab(best, tokens)
      self.bpe_merges.append(best)
      if verbose:
        print(f"Merged: {best} --> {''.join(best)}")

    vocab_set = set()
    for token in tokens:
      vocab_set.update(token.split())
    self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
    self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

  # encoding the function - convert the text to tokens and merge them up to reduce the total
  # token size. It returns the output ids
  def encode(self, text):
    words = text.strip().split()
    output_ids = []

    for word in words:
      tokens = self.get_vocab_tokens(word)
      while True:
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        merge_candidate = None
        for pair in pairs:
          if pair in self.bpe_merges:
            merge_candidate = pair
            break

        if not merge_candidate:
          break

        idx = pairs.index(merge_candidate)
        tokens = tokens[:idx] + [''.join(merge_candidate)] + tokens[idx + 2:]

      for token in tokens:
        output_ids.append(self.vocab[token])

      return output_ids

  # Decoding the tokens back to the text form.
  def decode(self, ids):
    tokens = [self.inverse_vocab[i] for i in ids]
    words = []
    word_bytes = bytearray()
    for token in tokens:
      if token == "</w>":
        words.append(word_bytes.decode("utf-8", errors="replace"))
        word_bytes = bytearray()
      elif token.startswith("<") and token.endswith(">"):
        byte_val = int(token[1:-1])
        word_bytes.append(byte_val)
      else:
        # merged byte tokens, split into chunks of <...>
        matches = re.findall(r"<\d+>", token)
      for m in matches:
        byte_val = int(m[1:-1])
        word_bytes.append(byte_val)

    return ' '.join(words)

# Example usage
with open("/content/taylorswift.txt", "r", encoding="utf-8") as f:
  text = f.read()

# train the tokenizer
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=100, verbose=True)

# encode the text
sample = "I remember it all too well."
encoded = tokenizer.encode(sample)
print("Encoded: ", encoded)

# decode back
decoded = tokenizer.decode(encoded)
print("Decoded: ", decoded)