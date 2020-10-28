"""Data utilities for the SMILE strings."""

from __future__ import division, print_function

from six.moves import range

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

LEN2_CHEM_ELEMENT = ["Cl", "Br"]

DEFAULT_REV_VOCAB = [
    'c', 'C', ')', '(', '1', '2', 'O', 'N', '[', ']', '@', 'H', 'n', '3', '=',
    '4', '+', 'F', 'S', 's', 'Cl', 'o', '5', '-', '#', 'Br', '/', '\\', '6',
    'I', '7', '8', '9', 'P', 'p', 'B', 'b'
]


class Vocabulary(object):
    """Interfaces to transform the SMILE tokens to initial continous tokens."""

    @classmethod
    def get_default_vocab(cls, with_at_symbol=False):
        """Return a default vocabulary object generated from Zinc dataset."""
        rev_vocab = list(DEFAULT_REV_VOCAB)
        if not with_at_symbol and "@" in rev_vocab:
            rev_vocab.remove('@')
        return cls(rev_vocab)

    @classmethod
    def load_vocab_from_file(cls, vocab_path):
        """Read vocab from old-style vocab file."""
        with open(vocab_path, "r") as fobj:
            vocab = []
            for _line in fobj:
                token = _line.strip()
                if token in _START_VOCAB:
                    # Skip the start tokens.
                    continue
                vocab.append(token)
        return cls(vocab)

    def __init__(self, rev_vocab):
        """Initialize a vocabulary for the SMILE representation.

        Args:
            rev_vocab: a list containing all the possible known tokens.
        """
        self._rev_vocab = _START_VOCAB + rev_vocab
        self._vocab = dict(zip(self._rev_vocab, range(len(self._rev_vocab))))
        # Set start symbol ids.
        for word in _START_VOCAB:
            setattr(self, "%s_ID" % word[1:], self._vocab[word])

    def query_token_id(self, token):
        """Return a token id by the token."""
        return self._vocab.get(token, self._vocab[_UNK])

    def query_token(self, token_id):
        """Return a token from its token_id."""
        if token_id < 0 or token_id > self._rev_vocab:
            raise KeyError(token_id)
        return self._rev_vocab[token_id]

    def __len__(self):
        """Return the size of the vocabulary.
        The size of vocabulary is determinted by the number of unique tokens."""
        return len(self._rev_vocab)


def true_smile_tokenizer(line, skip_at_symbol=True):
    """Return each character or atom as the token."""
    line = line.strip().replace(" ", "").replace("'", "").replace("\"", "")
    idx = 0
    tokens = []
    while idx < len(line):
        if idx < len(line) - 1 and line[idx:idx + 2] in LEN2_CHEM_ELEMENT:
            token = line[idx:idx + 2]
        else:
            token = line[idx]
        if not skip_at_symbol or token != "@":
            tokens.append(token)
        idx += len(token)
    return tokens


def sentence_to_token_ids(sentence,
                          vocabulary=Vocabulary.get_default_vocab(),
                          tokenizer=true_smile_tokenizer):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
        sentence: the sentence in bytes format to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
    Returns:
        a list of integers, the token-ids for the sentence.
    """
    return [vocabulary.query_token_id(w) for w in tokenizer(sentence)]
