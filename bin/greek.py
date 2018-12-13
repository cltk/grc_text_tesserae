from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize # Replace with CLTK
from nltk import pos_tag # Replace with CLTK

import codecs
import re

# Set pattern for finding documents in the CLTK Tesserae corpus
DOC_PATTERN = r'.*\.tess'

# Store tokenizers here for deployment below, i.e. set the defaults here so
# that the code below does not need to be updated
#
# Currently set with NLTK tokenizers/taggers for English
WORD_TOKENIZER = word_tokenize
SENT_TOKENIZER = sent_tokenize
POS_TAGGER = pos_tag

# WRITE DOCSTRING
class TesseraeCorpusReader(PlaintextCorpusReader):
    """
    """

    def docs(self, fileids):
        """
        Returns the complete text of a .tess file, closing the document after
        we are done reading it and yielding it in a memory-safe fashion.
        """

        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()


    def text(self, fileids):
        """
        Returns the text content of a .tess file, i.e. removing the bracketed
        citation info (e.g. "<Ach. Tat.  1.1.0>")
        """

        for doc in self.docs(fileids):
            doc = re.sub(r'<.+?>\s', '', doc) # Remove citation info
            doc = doc.rstrip() # Clean up final line breaks
            yield doc


    def paras(self, fileids):
        """
        Returns paragraphs in a .tess file, as defined by two \n characters.
        NB: Most .tess files do not have this feature; only the Homeric poems
        from what I have noticed so far. Perhaps a feature worth looking into.
        """

        for text in self.text(fileids):
            for para in text.split('\n\n'):
                yield para


    ### WRITE DOCSTRING
    def sents(self, fileids):
        """
        """

        for para in self.paras(fileids):
            for sent in SENT_TOKENIZER(para):
                yield sent


    ### WRITE DOCSTRING
    def words(self, fileids):
        """
        """
        for sent in self.sents(fileids):
            for token in WORD_TOKENIZER(sent):
                yield token


    def pos_tokenize(self, fileids):
        """
        Segments, tokenizes, and POS tag a document in the corpus.
        """

        for para in self.paras(fileids):
            yield [
                POS_TAGGER(WORD_TOKENIZER(sent))
                for sent in SENT_TOKENIZER(para)
            ]


tess_greek = TesseraeCorpusReader('texts', DOC_PATTERN)

if __name__ == "__main__":
    # sample = corpus.fileids()[0]
    sample = tess_greek.fileids()[0]
    s = tess_greek.pos_tokenize(sample)
    print(next(s))
