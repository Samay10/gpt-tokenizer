import regex as re

# gpt-4 split pattern - for more details, please refer: cl100k_base regex pattern
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# class that uses unicode encoding/decoding + Regex patterns + Model training: slightly advanced than the 
# BasicTokenizer.py file: uses cl100k_base regex pattern (as used in gpt4). This helps in capturing 's & 'S +
# spaces that are taken as one token and not multiple ones - like in tiktoken (gpt2).
class RegexTokenizer:
  def __init__(self, pattern=GPT4_SPLIT_PATTERN):
    self.pattern = re.compile(pattern)

  def split(self, text):
    return [t for t in self.pattern.findall(text) if t]

# use defaultdict to store the values as a dictionary and Counter that is used to
# store the frequency of pairs occurring in bpe_merges
from collections import defaultdict, Counter

# added functionality: special tokens (<|endoftext|>) - used to tell the model where to stop the 
# sentence. This makes training calculative.
class RegexBPETokenizer:
  def __init__(self, regex_tokenizer, special_tokens=None):
    self.regex_tokenizer = regex_tokenizer
    self.vocab = {}
    self.inverse_vocab = {}
    self.bpe_merges = []
    self.special_tokens = set(special_tokens) if special_tokens else set()
    self.special_token_pattern = (
        re.compile("|".join(re.escape(tok) for tok in self.special_tokens)) if self.special_tokens else None
    )

  def get_vocab_tokens(self, word):
    return list(word.encode("utf-8")) + [b"</w>"]   # byte level with end marker

  def get_stats(self, tokens):
    pairs = defaultdict(int)
    for token, freq in tokens.items():
      symbols = token.split()
      for i in range(len(symbols) - 1):
        pairs[(symbols[i], symbols[i+1])] += freq

    return pairs

  def merge_vocab(self, pair, tokens):
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    new_tokens = {}
    for token in tokens:
      new_token = pattern.sub("".join(pair), token)
      new_tokens[new_token] = tokens[token]

    return new_tokens

  def train(self, text, vocab_size, verbose=False):
    if self.special_token_pattern:
      parts = self.special_token_pattern.split(text)
      specials = self.special_token_pattern.findall(text)
      words = []
      for i, part in enumerate(parts):
        words.extend(self.regex_tokenizer.split(part))
        if i < len(specials):
          words.append(specials[i])
    else:
      words = self.regex_tokenizer.split(text)
      word_freq = Counter(words)

      tokens = {
          " ".join(map(str, self.get_vocab_tokens(word.decode("utf-8") if isinstance(word, bytes) else word))): freq
          for word, freq in word_freq.items()
      }

      while len(self.vocab) < vocab_size:
        pairs = self.get_stats(tokens)
        if not pairs:
          break
        
        best = max(pairs, key=pairs.get)
        tokens = self.merge_vocab(best, tokens)
        self.bpe_merges.append(best)

        if verbose:
          print(f"Merged: {best} --> {''.join(best)}")

    # build final vocab
    vocab_set = set()
    for token in tokens:
      vocab_set.update(token.split())
    for special in self.special_tokens:
      vocab_set.add(special)

    self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
    self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

  def encode(self, text, allowed_special=None):
    allowed = self.special_tokens if allowed_special == "all" else set()
    if self.special_token_pattern:
      parts = self.special_token_pattern.split(text)
      specials = self.special_token_pattern.findall(text)
    else:
      parts = [text]
      specials = []

    output_ids = []
    for i, part in enumerate(parts):
      if part:
        tokens = list(part.encode("utf-8")) + [b"</w>"]
        tokens = list(map(str, tokens))

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
          tokens = tokens[:idx] + ["".join(merge_candidate)] + tokens[idx + 2:]

        output_ids.extend([self.vocab[tok] for tok in tokens if tok in self.vocab])

      if i < len(specials):
        if specials[i] in allowed:
          output_ids.append(self.vocab[specials[i]])
        else:
          raise ValueError(f"Special token '{specials[i]}' not allowed. Pass allowed_special='all' to allow it.")

    return output_ids

  def decode(self, ids):
    tokens = [self.inverse_vocab[i] for i in ids]
    words = []
    current_word = bytearray()

    for tok in tokens:
      if tok in self.special_tokens:
        if current_word:
          words.append(current_word.decode("utf-8", errors="ignore"))
          current_word = bytearray()
        words.append(tok)
      elif tok == "b'</w>'":
        words.append(current_word.decode("utf-8", errors="ignore"))
        current_word = bytearray()
      elif tok.startswith("b'"):
        value = eval(tok)
        current_word.extend(value)
      else:
        try:
          current_word.extend(bytes([int(tok)]))
        except:
          pass

    if current_word:
      words.append(current_word.decode("utf-8", errors="ignore"))
    return " ".join(words)

  def bpe(self, mergeable_ranks, token, max_rank):
    # Convert string tokens to actual bytes
    parts = [bytes(eval(p)) if p.startswith("b'") else bytes([int(p)]) for p in token.split()]
    
    while True:
      min_idx = None
      min_rank = None
      for i, pair in enumerate(zip(parts[:-1], parts[1:])):
        merged = pair[0] + pair[1]
        rank = mergeable_ranks.get(merged)
        if rank is not None and (min_rank is None or rank < min_rank):
          min_idx = i
          min_rank = rank
      if min_rank is None or (max_rank is not None and min_rank >= max_rank):
        break
      parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    
    return parts

  def recover_merges(self):
    # Converts self.bpe_merges to rank-based byte merges
    mergeable_ranks = {}
    
    for rank, pair in enumerate(self.bpe_merges):
      # Convert strings back to raw bytes
      left = bytes(eval(pair[0])) if pair[0].startswith("b'") else bytes([int(pair[0])])
      right = bytes(eval(pair[1])) if pair[1].startswith("b'") else bytes([int(pair[1])])
      mergeable_ranks[left + right] = rank
    
    merges = {}
    for token, rank in mergeable_ranks.items():
      if len(token) == 1:
        continue
      
      pair = self.bpe(mergeable_ranks, " ".join(map(str, list(token))), max_rank=rank)
      assert len(pair) == 2
      ix0 = mergeable_ranks.get(pair[0])
      ix1 = mergeable_ranks.get(pair[1])
      if ix0 is not None and ix1 is not None:
        merges[(ix0, ix1)] = rank
    
    return merges
  
with open("/content/taylorswift.txt", "r", encoding="utf-8") as f:
  text = f.read()

regex_tokenizer = RegexTokenizer()
tokenizer = RegexBPETokenizer(regex_tokenizer)

tokenizer.train(text, vocab_size=100, verbose=True)
merges = tokenizer.recover_merges()

sample = "I remember it all too well."
encoded = tokenizer.encode(sample)
print("Encoded: ", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded: ", decoded)