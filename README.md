# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

Reflection:
Answer the following four (4) questions for Part 2 (Hidden Markov Models):

Q1: Give an example of a word which was correctly spelled by the user, but which was incorrectly “corrected” by the algorithm. Why did this happen?

One word I typed correctly was “Nevada”, and the algorithm changed it to “Nevade.” This happened because the Hidden Markov Model only relies on letter transition and emission probabilities learned from the training dataset. The model does not have any knowledge about which words are real or proper nouns. If another letter pattern appears more frequently in the data, the algorithm will choose that path, even when the original spelling is already correct. In this case, the model decided that the letter sequence ending in “de” was statistically more likely than the correct “da,” which caused the wrong correction.

Q2: Give an example of a word which was incorrectly spelled by the user, but which was still
incorrectly “corrected” by the algorithm. Why did this happen?

In the sentence I tested, I misspelled “abouy” (which was supposed to be “about”), and the model changed it to “aboly,” which is also incorrect. The reason this happened is that the model does not use a dictionary or any check to ensure that the output is a real word. It only picks the most likely sequence of letters based on its probability tables. Since the training data lacked examples of that specific typo, the model simply searched for a slightly more probable sequence, even though the result was still a nonsense word.

Q3: Give an example of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm. Why was this one correctly corrected, while the previous two were not?

After testing all the words from aspell.txt, I did not find any cases where my model actually corrected a spelling mistake to the true correct word. Instead, the model usually left the misspelling unchanged or replaced it with another incorrect form. One major reason for this is that the training data in aspell.txt does not provide enough real examples for the model to learn strong emission and transition probabilities for many common mistakes. The dataset has limited typo patterns, so the model does not develop a confident understanding of how to recover the correct spelling. Another reason is that my implementation does not use a dictionary or any scoring for valid English words. Because of that, the model has no way to tell the difference between a real correction and another random-looking letter sequence. With more diverse training data and a dictionary check built into the scoring process, the model would have a better chance of correctly fixing spelling errors.

Q4: How might the overall algorithm’s performance differ in the “real world” if that training dataset is taken from real typos collected from the internet, versus synthetic typos (programmatically generated)?

If the model learned from real typos that people actually make online, it would understand more natural mistakes. These could be things like pressing the wrong key, mixing up words that sound alike, repeating letters, or using the wrong capital letters. Learning from real examples would help the program fix everyday spelling mistakes more accurately. However, if the model only learned from fake typos made by a computer, it would not do as well. Those fake typos are usually too simple and follow the same patterns, like adding or removing random letters. A program trained only on those kinds of errors would work on easy mistakes but not on the real ones that people actually make. Training with real typos would make the spell fixer more accurate and useful.