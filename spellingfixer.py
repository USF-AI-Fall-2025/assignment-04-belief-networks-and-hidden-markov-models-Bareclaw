
import math
from collections import Counter, defaultdict
from pathlib import Path

# constants
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
START = "^"
END = "$"

# parameters
EMIT_K = 1.0
TRANS_K = 0.5
IDENTITY_WEIGHT = 0.2
INS_DEL_P = 0.9
WORD_BONUS = 2.0
NONWORD_PENALTY = 0.5
MAX_LEN_DIFF = 2
MAX_CANDIDATES = 3000



def laplace_logprobs(counts, alphabet, k):
    """Return log-probabilities with Laplace smoothing over an alphabet."""
    total = sum(counts.values()) + k * len(alphabet)
    probs = {}
    for letter in alphabet:
        count = counts.get(letter, 0) + k
        probs[letter] = math.log(count / total)
    return probs


def add_emissions(emit_counts, correct, observed, weight):
    """Accumulate emission counts for lettert to letter pairs with a given weight."""
    for g, o in zip(correct.lower(), observed.lower()):
        if g.isalpha() and o.isalpha():
            emit_counts[g][o] += weight



def train_from_aspell(filename):
    """
    Read aspell.txt (a list of correct words and typos) and learn from it.
    Example of aspell-like data:
        definitely: definetly definately
        weird: wierd
    Returns emission, transition, and dictionary data.
    """
    file_path = Path(filename)
    if not file_path.exists():
        print("Error: aspell.txt not found.")
        return None, None, set()

    # create emission and transition count structures manually
    emit_counts = {}
    for c in LETTERS:
        emit_counts[c] = defaultdict(float)

    trans_counts = {START: Counter()}
    for c in LETTERS:
        trans_counts[c] = Counter()

    dictionary = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            correct, rest = line.split(":", 1)
            correct = correct.strip().lower()
            typos = []
            for word in rest.split():
                word = word.strip().lower()
                if word:
                    typos.append(word)

            # only letters in word
            word = ""
            for ch in correct:
                if ch.isalpha():
                    word += ch
            if not word:
                continue

            dictionary.add(word)

            # transitions
            trans_counts[START][word[0]] += 1
            for i in range(len(word) - 1):
                a = word[i]
                b = word[i + 1]
                trans_counts[a][b] += 1
            trans_counts[word[-1]][END] += 1

            # emissions
            add_emissions(emit_counts, correct, correct, IDENTITY_WEIGHT)
            for t in typos:
                add_emissions(emit_counts, correct, t, 1.0)

    # build log probabilities
    emit_log = {}
    for c in LETTERS:
        emit_log[c] = laplace_logprobs(emit_counts[c], LETTERS, EMIT_K)

    trans_log = {}
    trans_log[START] = laplace_logprobs(trans_counts[START], LETTERS, TRANS_K)
    for c in LETTERS:
        trans_log[c] = laplace_logprobs(trans_counts[c], LETTERS + [END], TRANS_K)

    return emit_log, trans_log, dictionary

def get_negative_infinity_value():
    """Just returns a very small number used to represent 'impossible' scores."""
    return -float("inf")

def viterbi_decode(word, emit_log, trans_log):
    """Guess what the correct spelling is using the Viterbi algorithm."""
    word = word.lower()
    if not word.isalpha():
        return word

    n = len(word)
    V = []
    back = []
    for _ in range(n):
        V.append(defaultdict(lambda: -float("inf")))
        back.append({})

    # initialize
    for s in LETTERS:
        trans_prob = trans_log[START].get(s, float('-inf'))
        emit_prob = emit_log[s].get(word[0], float('-inf'))
        V[0][s] = trans_prob + emit_prob
        back[0][s] = None

    # main loop
    for i in range(1, n):
        for s in LETTERS:
            best_prev = None
            best_val = -float("inf")
            for p in LETTERS:
                prev_score = V[i - 1][p]
                trans_prob = trans_log[p].get(s, float('-inf'))
                emit_prob = emit_log[s].get(word[i], float('-inf'))
                val = prev_score + trans_prob + emit_prob
                if val > best_val:
                    best_val = val
                    best_prev = p
            V[i][s] = best_val
            back[i][s] = best_prev

    # termination
    best_last = None
    best_score = -float("inf")
    for s in LETTERS:
        score = V[-1][s] + trans_log[s].get(END, -float("inf"))
        if score > best_score:
            best_score = score
            best_last = s

    # backtrack
    decoded = [best_last]
    for i in range(n - 1, 0, -1):
        decoded.append(back[i][decoded[-1]])
    decoded.reverse()
    return "".join(decoded)



def score_word(true_word, observed, emit_log, trans_log):
    """Give a numeric score for how likely the true_word explains the observed one."""
    t = true_word.lower()
    o = observed.lower()
    if not t.isalpha():
        return -float("inf")

    score = trans_log[START].get(t[0], -float("inf"))
    for i in range(len(t) - 1):
        a = t[i]
        b = t[i + 1]
        score += trans_log[a].get(b, -float("inf"))
    score += trans_log[t[-1]].get(END, -float("inf"))

    for g, ob in zip(t, o):
        score += emit_log[g].get(ob, -float("inf"))

    if len(t) != len(o):
        score += abs(len(t) - len(o)) * math.log(INS_DEL_P)
    return score



def get_candidates(dictionary, observed):
    """Get dictionary words that are about the same length as the typo."""
    o = observed.lower()
    cands = []
    for w in dictionary:
        if abs(len(w) - len(o)) <= MAX_LEN_DIFF:
            cands.append(w)
    if len(cands) > MAX_CANDIDATES:
        cands = cands[:MAX_CANDIDATES]
    return cands



def fix_word(word, emit_log, trans_log, dictionary, use_dict=True):
    """Fix one word using the model, and check dictionary words if needed."""
    if not word.isalpha():
        return word

    wl = word.lower()
    guess = viterbi_decode(wl, emit_log, trans_log)
    best_word = guess
    best_score = score_word(guess, wl, emit_log, trans_log)

    if guess in dictionary:
        best_score += WORD_BONUS
    else:
        best_score -= NONWORD_PENALTY

    if use_dict:
        candidates = get_candidates(dictionary, wl)
        for cand in candidates:
            s = score_word(cand, wl, emit_log, trans_log) + WORD_BONUS
            if s > best_score:
                best_word = cand
                best_score = s

    return match_case(word, best_word)



def match_case(original, fixed):
    """Keep the same capitalization as the original word."""
    if original.isupper():
        return fixed.upper()
    elif original[0].isupper():
        return fixed.capitalize()
    else:
        return fixed.lower()



def fix_line(line, emit_log, trans_log, dictionary, use_dict=True):
    """Fix all words in a line (sentence)."""
    words = line.split()
    fixed_words = []
    for w in words:
        fixed_words.append(fix_word(w, emit_log, trans_log, dictionary, use_dict))
    return " ".join(fixed_words)



def main():
    """Run the program, learn from aspell.txt, and let the user type sentences to fix."""
    aspell_path = "aspell.txt"  # default file next to script
    emit_log, trans_log, dictionary = train_from_aspell(aspell_path)
    if emit_log is None:
        return  # aspell.txt missing; already warned
    
    use_dict = True  # keep dictionary rescue on by default
    print("Spelling Fixer (HMM) — type text and press Enter. Blank line or Ctrl+C to quit.")
    try:
        while True:
            text = input("Enter some text: ").strip()
            if not text:
                break
            fixed = fix_line(text, emit_log, trans_log, dictionary, use_dict=use_dict)
            print("Corrected text:", fixed)
    except KeyboardInterrupt:
        print("\nDone!")


if __name__ == "__main__":
    main()






