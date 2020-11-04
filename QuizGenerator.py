import argparse
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.okt = Okt()
        self.stopwords = ['중인' ,'만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자","아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가",]

    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(len(sentences)):
            if len(sentences[idx]) <= 10:  # if length of current sentence is lte 10, concat next sentence with current sentence.
                sentences[idx - 1] += (" " + sentences[idx])
                sentences[idx] = ""
    
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not "":
                nouns.append(" ".join([noun for noun in self.okt.nouns(str(sentence)) if noun not in self.stopwords and len(noun) > 1]))
    
        return nouns

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
  
      # 명사로 이루어진 문장을 입력받아 sklearn의 TfidfVectorizer.fit_transform을 이용하여 tfidf matrix를 만든 후 Sentence graph를 return 한다.
      # Used in sentence summarization.
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()  # Sentence-Term Matrix (derived from DTM, Document-Term Matrix)
        self.graph_sentence = np.matmul(tfidf_mat, tfidf_mat.T)  # Correlation Matrix among sentences.
        return self.graph_sentence

      # 명사로 이루어진 문장을 입력받아 sklearn의 CountVectorizer.fit_transform을 이용하여 matrix를 만든 후 word graph와 {idx: word}형태의 dictionary를 return한다.
      # Used in keyword extraction.
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.matmul(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}

# TODO: Understand this math stuffs...
class Rank(object):
    def get_ranks(self, graph, d=0.85):  # input: Sentence-Term Matrix (square matrix)
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0  # make diagonal element to zero
            link_sum = np.sum(A[:, id])  # A[:, id] means id-th column vector
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = B
        return {idx: r[0] for idx, r in enumerate(ranks)}

class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()  # What we've created
        self.sentences = self.sent_tokenize.text2sentences(text)
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)

        self.graph_matrix = GraphMatrix()  # Also what we've created
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)  # Build sentence graph (adjacency matrix) with given nouns
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)  # Build words graph(?) with given nouns

        self.rank = Rank()  # Also what we've created
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    def summarize(self, sent_num=3):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()

        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    def keywords(self, word_num=10):
        keywords = []
        index = []
        for idx in self.sorted_word_rank_idx[:word_num]:
            index.append(idx)
    
        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords

if __name__ == '__main__':
    # TODO: Clarify description.
    parser = argparse.ArgumentParser(description='Generate quiz from given text.')
    parser.add_argument('text', type=str, help='text')
    parser.add_argument('quiz_cnt', type=int, help='quiz_cnt')

    args = parser.parse_args()
    print(args.text)
    print(args.quiz_cnt)

    text, quiz_cnt = args

    print(text)

    kkma = Kkma()

    

