import math
from collections import Counter, defaultdict
from pathlib import Path

# constants
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
START = "^"
END = "$"

# smoothing params
EMIT_K = 1.0
TRANS_K = 0.5

# training weights
IDENTITY_WEIGHT = 0.2  # count a little mass for correct->correct pairs

def laplace_logprobs(counts, alphabet, k):
    """Return log-probabilities with Laplace smoothing over an alphabet."""
    total = sum(counts.values()) + k * len(alphabet)
    probs = {}
    for letter in alphabet:
        count = counts.get(letter, 0) + k
        probs[letter] = math.log(count / total)
    return probs

def add_emissions(emit_counts, correct, observed, weight):
    """Accumulate emission counts for letter-to-letter pairs with a given weight."""
    for g, o in zip(correct.lower(), observed.lower()):
        if g.isalpha() and o.isalpha():
            emit_counts[g][o] += weight

def train_from_aspell(filename):
    """
    Read aspell.txt (a list of correct words and typos) and learn from it.

    Example:
        definitely: definetly definately
        weird: wierd

    Returns (emit_log, trans_log).
    """
    file_path = Path(filename)
    if not file_path.exists():
        print("Error: aspell.txt not found.")
        return None, None

    # emission counts for each gold letter -> observed letter
    emit_counts = {c: defaultdict(float) for c in LETTERS}

    # letter bigram transition counts over gold letters
    trans_counts = {START: Counter()}
    for c in LETTERS:
        trans_counts[c] = Counter()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            correct, rest = line.split(":", 1)
            correct = correct.strip().lower()

            # keep only letters in the "correct" word
            word = "".join(ch for ch in correct if ch.isalpha())
            if not word:
                continue

            # transitions from ^ to first, internal bigrams, and last to $
            trans_counts[START][word[0]] += 1
            for i in range(len(word) - 1):
                trans_counts[word[i]][word[i + 1]] += 1
            trans_counts[word[-1]][END] += 1

            # emissions: give identity pairs small weight
            add_emissions(emit_counts, correct, correct, IDENTITY_WEIGHT)

            # emissions: each listed typo gets full weight
            typos = [w.strip().lower() for w in rest.split() if w.strip()]
            for t in typos:
                add_emissions(emit_counts, correct, t, 1.0)

    # convert counts to log-probabilities
    emit_log = {c: laplace_logprobs(emit_counts[c], LETTERS, EMIT_K) for c in LETTERS}
    trans_log = {}
    trans_log[START] = laplace_logprobs(trans_counts[START], LETTERS, TRANS_K)
    for c in LETTERS:
        trans_log[c] = laplace_logprobs(trans_counts[c], LETTERS + [END], TRANS_K)

    return emit_log, trans_log

def viterbi_decode(word, emit_log, trans_log):
    """Decode using pure Viterbi algorithm - no dictionary lookup."""
    w = word.lower()
    if not w.isalpha():
        return word

    n = len(w)
    V = [defaultdict(lambda: -float("inf")) for _ in range(n)]
    back = [{} for _ in range(n)]

    # initialize: start state to first letter
    for s in LETTERS:
        V[0][s] = trans_log[START].get(s, float("-inf")) + emit_log[s].get(w[0], float("-inf"))
        back[0][s] = None

    # dynamic programming: fill in the trellis
    for i in range(1, n):
        for s in LETTERS:
            best_prev, best_val = None, -float("inf")
            emit = emit_log[s].get(w[i], float("-inf"))
            for p in LETTERS:
                val = V[i - 1][p] + trans_log[p].get(s, float("-inf")) + emit
                if val > best_val:
                    best_val, best_prev = val, p
            V[i][s] = best_val
            back[i][s] = best_prev

    # termination: find best final state
    best_last, best_score = None, -float("inf")
    for s in LETTERS:
        score = V[-1][s] + trans_log[s].get(END, float("-inf"))
        if score > best_score:
            best_score, best_last = score, s

    # backtrack to get the decoded sequence
    decoded = [best_last]
    for i in range(n - 1, 0, -1):
        decoded.append(back[i][decoded[-1]])
    decoded.reverse()
    return "".join(decoded)

def fix_word(word, emit_log, trans_log):
    """
    Pure Viterbi decoding only - return the HMM result directly.
    No dictionary lookup or candidate comparison.
    """
    if not word.isalpha():
        return word
    
    # Get the pure Viterbi result
    guess = viterbi_decode(word, emit_log, trans_log)
    
    # Preserve simple case patterns
    if word.isupper():
        return guess.upper()
    if word.istitle():
        return guess.title()
    return guess

def fix_line(line, emit_log, trans_log):
    """Fix all tokens in a line using pure Viterbi."""
    return " ".join(fix_word(tok, emit_log, trans_log) for tok in line.split())

def main():
    """Train from aspell.txt, then decode user input with pure Viterbi only."""
    aspell_path = "aspell.txt"
    emit_log, trans_log = train_from_aspell(aspell_path)
    if emit_log is None:
        return

    print("Spelling Fixer (Pure Viterbi) - type text and press Enter.")
    print("Blank line or Ctrl+C to quit.")
    try:
        while True:
            text = input("Enter some text: ").strip()
            if not text:
                break
            fixed = fix_line(text, emit_log, trans_log)
            print("Corrected text:", fixed)
    except KeyboardInterrupt:
        print("\nDone!")

if __name__ == "__main__":
    main()