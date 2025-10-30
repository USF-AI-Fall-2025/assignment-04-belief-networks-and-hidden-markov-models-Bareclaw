# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

Reflection:
Answer the following four (4) questions for Part 2 (Hidden Markov Models):

Q1: Give an example of a word which was correctly spelled by the user, but which was incorrectly “corrected” by the algorithm. Why did this happen?

The correct word that I typed was "Nevada", and the corrected word was "Have". This happened because even though “Nevada” is a real word in the dictionary, the model gives a higher score to “Have.” The Hidden Markov Model looks at how likely letters are to appear next to each other, and “Have” has more common letter patterns than “Nevada.” The scoring system also gives a small bonus to real words and a small penalty to words that seem uncommon. When those numbers combine, the program sometimes chooses a different word that is more common even when the original word was already correct.

Q2: Give an example of a word which was incorrectly spelled by the user, but which was still
incorrectly “corrected” by the algorithm. Why did this happen?

The incorrect word that I typed was "definately", and the corrected word was "definite". This error happened because both “definitely” and “definite” are in the training data, and the typo “definately” looks similar to both. The model compares letter patterns and their probabilities, but the penalty for missing or extra letters is very small. That means dropping the last “ly” barely hurts the score. Since “definite” is shorter and fits the pattern well, the program sometimes picks it instead of “definitely". This shows that weak penalties for length changes can make the program prefer the wrong, shorter word. 

Q3: Give an example of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm. Why was this one correctly corrected, while the previous two were not?

The misspelled word that I used was abilitey, and the corrected word was ability. This worked because “abilitey” and “ability” are already linked in the training data. The model learned how the wrong letters in “abilitey” usually match to the correct ones in “ability.” Also, “ability” is in the dictionary, so it gets a score bonus, and its letter patterns are common and easy for the model to predict. Together, those factors helped the program choose the right word confidently.

Q4: How might the overall algorithm’s performance differ in the “real world” if that training dataset is taken from real typos collected from the internet, versus synthetic typos (programmatically generated)?

If the model learned from real typos that people actually make online, it would understand more natural mistakes. These could be things like pressing the wrong key, mixing up words that sound alike, repeating letters, or using the wrong capital letters. Learning from real examples would help the program fix everyday spelling mistakes more accurately. However, if the model only learned from fake typos made by a computer, it would not do as well. Those fake typos are usually too simple and follow the same patterns, like adding or removing random letters. A program trained only on those kinds of errors would work on easy mistakes but not on the real ones that people actually make. Training with real typos would make the spell fixer more accurate and useful.