from perturbers.input_perturbers import *

# Test AddLetterPerturber

def test_add_letter_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert AddLetterPerturber(1, 0).perturb(text) == text

def test_add_letter_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago. Behold!"
    assert AddLetterPerturber(1, 1).perturb(text) == "Tio tnhe poobnl, I rwenth 345 minutswes aguo. Behuolde!"

def test_add_letter_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(AddLetterPerturber(42, 0.5).perturb(text))
    assert AddLetterPerturber(42, 0.5).perturb(text) == "To hthe pool, I wevnt 345 minutes adgo. Irt was greadth ! H"

def test_add_letter_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(AddLetterPerturber(2, 0.25).perturb(text))
    assert AddLetterPerturber(2, 0.25).perturb(text) == "To the pojol, I went 345 minutes ago. It was great ! H. . . "

def test_add_letter_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(AddLetterPerturber(2, 0.75).perturb(text))
    assert AddLetterPerturber(2, 0.75).perturb(text) == "To the pojol, I went 345 minutwes ago. Ite was great ! H. . . "

# Test AddWhitespaceBeforePunctuationPerturber

def test_add_whitespace_before_punctuation_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert AddWhitespaceBeforePunctuationPerturber(1, 0).perturb(text) == text

def test_add_whitespace_before_punctuation_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago. Behold!"
    assert AddWhitespaceBeforePunctuationPerturber(1, 1).perturb(text) == text.replace(".", " .").replace(",", " ,").replace("!", " !")

def test_add_whitespace_before_punctuation_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(AddWhitespaceBeforePunctuationPerturber(42, 0.5).perturb(text))
    assert AddWhitespaceBeforePunctuationPerturber(42, 0.5).perturb(text) == "To the pool, I went 345 minutes ago . It was great  ! H"

def test_add_whitespace_before_punctuation_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(AddWhitespaceBeforePunctuationPerturber(2, 0.25).perturb(text))
    assert AddWhitespaceBeforePunctuationPerturber(2, 0.25).perturb(text) == "To the pool, I went 345 minutes ago. It was great  ! H . . . "

def test_add_whitespace_before_punctuation_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(AddWhitespaceBeforePunctuationPerturber(2, 0.75).perturb(text))
    assert AddWhitespaceBeforePunctuationPerturber(2, 0.75).perturb(text) == "To the pool, I went 345 minutes ago. It was great  ! H . .  . "

# Test DropPunctuationPerturber

def test_drop_punctuation_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert DropPunctuationPerturber(1, 0).perturb(text) == text

def test_drop_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago. Behold!"
    assert DropPunctuationPerturber(1, 1).perturb(text) == text.replace(".", "").replace(",", "").replace("!", "")

def test_drop_punctuation_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    assert DropPunctuationPerturber(42, 0.5).perturb(text) == "To the pool, I went 345 minutes ago It was great  H"

def test_drop_punctuation_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropPunctuationPerturber(2, 0.25).perturb(text))
    assert DropPunctuationPerturber(2, 0.25).perturb(text) == "To the pool, I went 345 minutes ago. It was great  H . . "

def test_drop_punctuation_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropPunctuationPerturber(2, 0.75).perturb(text))
    assert DropPunctuationPerturber(2, 0.75).perturb(text) == "To the pool, I went 345 minutes ago. It was great  H .  "

# Test DropLetterPerturber

def test_drop_letter_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert DropLetterPerturber(1, 0).perturb(text) == text

def test_drop_letter_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    print(DropLetterPerturber(1, 1).perturb(text))
    assert DropLetterPerturber(1, 1).perturb(text) == "To the poo, I ent 345 inutes ao."

def test_drop_letter_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(DropLetterPerturber(42, 0.5).perturb(text))
    assert DropLetterPerturber(42, 0.5).perturb(text) == "To the ool, I went 345 minutes ao. It was great ! H"

def test_drop_letter_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropLetterPerturber(2, 0.25).perturb(text))
    assert DropLetterPerturber(2, 0.25).perturb(text) == "To the poo, I went 345 minutes ago. It was great ! H. . . "

def test_drop_letter_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropLetterPerturber(2, 0.75).perturb(text))
    assert DropLetterPerturber(2, 0.75).perturb(text) == "To the poo, I nt 345 miutes ago. It was grea ! H. . . "

# Test DropWhitespacePerturber

def test_drop_whitespace_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert DropWhitespacePerturber(1, 0).perturb(text) == text

def test_drop_whitespace_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    assert DropWhitespacePerturber(1, 1).perturb(text) == text.replace(" ", "")

def test_drop_whitespace_perturber_half():
    text = "To the pool, I went 345 minutes ago."
    assert DropWhitespacePerturber(42, 0.5).perturb(text) == "To thepool,Iwent 345 minutes ago."

# Test DropWhitespaceAroundPunctuationPerturber

def test_drop_whitespace_punctuation_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert DropWhitespaceAroundPunctuationPerturber(1, 0).perturb(text) == text

def test_drop_whitespace_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    assert DropWhitespaceAroundPunctuationPerturber(1, 1).perturb(text) == text.replace(", ", ",")

def test_drop_whitespace_punctuation_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    assert DropWhitespaceAroundPunctuationPerturber(42, 0.5).perturb(text) == "To the pool, I went 345 minutes ago.It was great!H"

def test_drop_whitespace_punctuation_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropWhitespaceAroundPunctuationPerturber(2, 0.25).perturb(text))
    assert DropWhitespaceAroundPunctuationPerturber(2, 0.25).perturb(text) == "To the pool, I went 345 minutes ago. It was great!H. . . "

def test_drop_whitespace_punctuation_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(DropWhitespaceAroundPunctuationPerturber(2, 0.75).perturb(text))
    assert DropWhitespaceAroundPunctuationPerturber(2, 0.75).perturb(text) == "To the pool, I went 345 minutes ago. It was great!H..."

# Test LetterCasePerturber

def test_letter_case_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert LetterCasePerturber(1, 0).perturb(text) == text

def test_letter_case_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    assert LetterCasePerturber(1, 1).perturb(text) == text.swapcase()

def test_letter_case_perturber_half():
    text = "To the pool, I went 345 minutes ago."
    assert LetterCasePerturber(42, 0.5).perturb(text) == "TO The POOL, i wEnt 345 mINUtes agO."

# Test RepeatLetterPerturber

def test_repeat_letter_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert RepeatLetterPerturber(1, 0).perturb(text) == text

def test_repeat_letter_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    print(RepeatLetterPerturber(1, 1).perturb(text))
    assert RepeatLetterPerturber(1, 1).perturb(text) == "TTTo ttthe ppooool, I wwweent 345 mmiiinnutes aaago."

def test_repeat_letter_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(RepeatLetterPerturber(42, 0.5).perturb(text))
    assert RepeatLetterPerturber(42, 0.5).perturb(text) == "Too tthe pooool, I wennnt 345 mmmiinuuutes aaago. It wasss greeeat ! H"

def test_repeat_letter_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(RepeatLetterPerturber(2, 0.25).perturb(text))
    assert RepeatLetterPerturber(2, 0.25).perturb(text) == "To ttthe pool, I wenttt 345 minutes ago. It waas gggrreat ! H. . . "

def test_repeat_letter_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(RepeatLetterPerturber(2, 0.75).perturb(text))
    assert RepeatLetterPerturber(2, 0.75).perturb(text) == "To ttthe poooool, I weeennt 345 mmmiinnutes aaago. Ittt waaas ggreeat ! H. . . "

# Test RepeatPunctuationPerturber

def test_repeat_punctuation_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert RepeatPunctuationPerturber(1, 0).perturb(text) == text

def test_repeat_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago."
    print(RepeatPunctuationPerturber(1, 1).perturb(text))
    assert RepeatPunctuationPerturber(1, 1).perturb(text) == "To the pool,, I went 345 minutes ago..."

def test_repeat_punctuation_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(RepeatPunctuationPerturber(42, 0.5).perturb(text))
    assert RepeatPunctuationPerturber(42, 0.5).perturb(text) == "To the pool, I went 345 minutes ago... It was great !! H"

def test_repeat_punctuation_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(RepeatPunctuationPerturber(2, 0.25).perturb(text))
    assert RepeatPunctuationPerturber(2, 0.25).perturb(text) == "To the pool, I went 345 minutes ago. It was great !! H. ... . "

def test_repeat_punctuation_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(RepeatPunctuationPerturber(2, 0.75).perturb(text))
    assert RepeatPunctuationPerturber(2, 0.75).perturb(text) == "To the pool, I went 345 minutes ago. It was great !! H.. ... .. "

# Test SubstituteLetterPerturber

def test_substitute_letter_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert SubstituteLetterPerturber(1, 0).perturb(text) == text

def test_substitute_letter_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago. Behold!"
    assert SubstituteLetterPerturber(1, 1).perturb(text) == "To jhe kool, I wenu 345 minihes ano. Beholq!"

def test_substitute_letter_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(SubstituteLetterPerturber(42, 0.5).perturb(text))
    assert SubstituteLetterPerturber(42, 0.5).perturb(text) == "To ths kuol, I wevt 345 mibutes xgo. Yt wss great ! H"

def test_substitute_letter_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(SubstituteLetterPerturber(2, 0.25).perturb(text))
    assert SubstituteLetterPerturber(2, 0.25).perturb(text) == "To the ppul, I went 345 minutes ago. It was great ! H. . . "

def test_substitute_letter_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(SubstituteLetterPerturber(2, 0.75).perturb(text))
    assert SubstituteLetterPerturber(2, 0.75).perturb(text) == "To the ppul, I wekt 345 minutes cgo. Lt wds great ! H. . . "

# Test SubstituteLetterPerturber

def test_swap_adj_letter_perturber_0():
    text = "To the pool, I went 345 minutes ago."
    assert SwapAdjacentLetterPerturber(1, 0).perturb(text) == text

def test_swap_adj_letter_punctuation_perturber_1():
    text = "To the pool, I went 345 minutes ago. Behold!"
    assert SwapAdjacentLetterPerturber(1, 1).perturb(text) == "Ot teh opol, I wetn 345 miuntes aog. Behodl!"

def test_swap_adj_letter_perturber_half():
    text = "To the pool, I went 345 minutes ago. It was great ! H"
    print(SwapAdjacentLetterPerturber(42, 0.5).perturb(text))
    assert SwapAdjacentLetterPerturber(42, 0.5).perturb(text) == "To hte opol, I ewnt 345 imnutes ago. Ti aws great ! H"

def test_swap_adj_letter_perturber_quarter():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(SwapAdjacentLetterPerturber(2, 0.25).perturb(text))
    assert SwapAdjacentLetterPerturber(2, 0.25).perturb(text) == "To the opol, I went 345 minutes ago. Ti was great ! H. . . "

def test_swap_adj_letter_perturber_three_quarters():
    text = "To the pool, I went 345 minutes ago. It was great ! H. . . "
    print(SwapAdjacentLetterPerturber(2, 0.75).perturb(text))
    assert SwapAdjacentLetterPerturber(2, 0.75).perturb(text) == "To the opol, I wetn 345 miuntes aog. Ti aws graet ! H. . . "