# -*- coding: utf-8 -*-
"""TextRank_newspaper_sentence_summarization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Dogdriip/notebooks/blob/main/TextRank_QuizGenerator.ipynb
"""

!pip install konlpy

from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

kkma = Kkma()
print(kkma.pos("안녕하세요? 오늘 날씨가 참 좋네요."))

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

text = """
아들의 군 휴가 특혜 의혹과 관련해 검찰로부터 무혐의 처분을 받은 추미애 법무부 장관이 30일 "제보자의 일방적인 주장을 검증하지 않고 단지 정쟁의 도구로 삼은 무책임한 세력들은 반드시 엄중한 책임을 져야 한다"며 경고하고 나섰다. 이에 국민의힘은 "적반하장에 기가 찰 노릇"이라며 반박했다.
추 장관이 이날 오전 8시쯤 페이스북에 국민에게 사과의 뜻을 밝히면서도 야권과 언론을 향해 유감을 표명했다. 검찰의 무혐의 처분 이후 이틀 만에 반격에 나선 것이다.
추 장관은 아들 서모씨(27)의 군 휴가 특혜 관련 의혹에 대해 "정치공세의 성격이 짙은 무리한 고소·고발로 국론을 분열시키고 국력을 소모한 사건"으로 규정하고, "무책임한 세력들은 반드시 엄중한 책임을 져야 할 것이다. 합당한 사과가 없을 시 후속 조치를 취하겠다"고 경고했다.
언론을 향해서도 "보도 양태에 깊은 유감을 보내지 않을 수 없다"며 "사실과 진실을 짚는 대신 허위의 주장을 그대로 싣고 나아가 허위를 사실인 양 보도한 다수 언론은 국민에게 커다란 실망과 상처를 줬다"고 말했다.
'무책임한 세력'이 누구인지는 직접 밝히진 않았지만 국민의힘 등 야권과 보수 언론을 겨냥한 것으로 풀이된다.
이에 장제원 국민의힘 의원은 자신의 페이스북에 "'방귀 낀 X이 성낸다'라는 말이 있다. 추 장관의 적반하장에 기가 찰 노릇"이라고 밝혔다.
장 의원은 이어 "추 장관이 수사 관련 자료가 공개돼 자신의 거짓말이 탄로가 나자, 사과는 커녕 국민과 언론을 향해 겁박까지 하고 나섰다"고 지적했다.
장 의원은 또 "'반드시 엄중한 책임을 져야 한다' 라구요"라고 반문한 뒤 "국민 앞에서 눈 하나 깜짝하지 않고 했던 거짓말부터 엄중한 책임을 져야 할 것"이라고 반박했다.
그러면서 "추 장관은 '합당한 사과가 없을 시 후속 조치를 취할 것' 이라며 협박도 서슴지 않는다"며 "저희들이 하고 싶은 말이고, 추 장관이 했던 거짓말에 대해 합당한 사과가 없을 시, 국민과 함께 후속조치를 취해 나갈 것"이라고 경고했다.
국민의힘 서울 송파병 당협위원장을 맡고 있는 김근식 경남대 교수도 이날 페이스북에 "추 장관님이 국민에게 거짓말한 것부터 사과해야 한다"며 "먼저 죄없는 젊은이를 거짓말쟁이로 낙인찍은 것을 사과해야 한다"고 지적했다.
또 추 장관이 '검찰개혁 완수'를 언급한 것과 관련해선 "제발 이제는 검찰개혁이란 말 좀 그만하라"며 "국민들은 이제 검찰개혁이라 쓰고 '검찰 길들이기'라고 읽는다"고 덧붙였다.
안혜진 국민의당 논평에서 "부정과 부조리, 비상식적인 짓을 해도 내 편이기만 하면 무조건 보호받는 나라가 대통령께서 꿈꾸었던 나라는 아닐 것"이라고 말했다.
"""

textrank = TextRank(text)
for row in textrank.summarize():
  print(row)
  print()

textrank.keywords()

text = """
사회주의란 말은 다음 다섯 가지의 각기 다른 뜻으로 사용되고 있다.
① 생산수단의 사회적 소유와 계획경제 제도를 수단으로, 자유·평등·사회정의를 실현할 것을 주장하는 사상과 운동을 뜻하는 경우(고전적 사회주의의 뜻으로 사용되는 경우)
② 생산수단의 사회적 소유와 계획경제라고 하는 제도 자체만을 가리켜 뜻하는 경우
③ 사회주의의 목적만을 가리키는 경우(자본주의보다 한층 훌륭한 사회를 뜻하는 경우)
④ 공산주의의 첫째 단계 또는 보다 낮은 단계를 뜻하는 경우(공산주의자 특유의 반논리적 용법)
⑤ 민주사회주의적 용법(민주주의적 방법에 의하여 민주주의 자체를 완성함으로써 사회를 개조하려는 사상 및 운동 또는 민주주의의 최고의 형태를 뜻하는 경우) 등이다.
"""

textrank = TextRank(text)
textrank.keywords()

text = """
미안하다 이거 보여주려고 어그로끌었다. 나루토 사스케 싸움수준 ㄹㅇ실화냐? 진짜 세계관최강자들의 싸움이다. 그찐따같던 나루토가 맞나? 진짜 나루토는 전설이다.진짜옛날에 맨날나루토봘는데 왕같은존재인 호카게 되서 세계최강 전설적인 영웅이된나루토보면 진짜내가다 감격스럽고 나루토 노래부터 명장면까지 가슴울리는장면들이 뇌리에 스치면서 가슴이 웅장해진다. 그리고 극장판 에 카카시앞에 운석날라오는 거대한 걸 사스케가 갑자기 순식간에 나타나서 부숴버리곤 개간지나게 나루토가 없다면 마을을 지킬 자는 나밖에 없다 라며 바람처럼 사라진장면은 진짜 나루토처음부터 본사람이면 안울수가없더라 진짜 너무 감격스럽고 보루토를 최근에 알았는데 미안하다. 지금20화보는데 진짜 나루토세대나와서 너무 감격스럽고 모두어엿하게 큰거보니 내가 다 뭔가 알수없는 추억이라해야되나 그런감정이 이상하게 얽혀있다. 시노는 말이많아진거같다 좋은선생이고.그리고 보루토왜욕하냐 귀여운데 나루토를보는것같다 성격도 닮았어 그리고버루토에 나루토사스케 둘이싸워도 이기는 신같은존재 나온다는게 사실임?? 그리고인터닛에 쳐봣는디 이거 ㄹㅇㄹㅇ 진짜팩트냐? 저적이 보루토에 나오는 신급괴물임?ㅡ 나루토사스케 합체한거봐라 진짜 ㅆㅂ 이거보고 개충격먹어가지고 와 소리 저절로 나오더라 ;; 진짜 저건 개오지는데. 저게 ㄹㅇ이면 진짜 꼭봐야돼 진짜 세계도 파괴시키는거아니야 . 와 진짜 나루토사스케가 저렇게 되다니 진짜 눈물나려고했다. 버루토그라서 계속보는중인데 저거 ㄹㅇ이냐? 하. ㅆㅂ 사스케 보고싶다.  진짜언제 이렇게 신급 최강들이 되었을까 옛날생각나고 나 중딩때생각나고 뭔가 슬프기도하고 좋기도하고 감격도하고 여러가지감정이 복잡하네. 아무튼 나루토는 진짜 애니중최거명작임.
"""

textrank = TextRank(text)
for row in textrank.summarize():
  print(row)
  print()
print(textrank.keywords())

def mask(text: str, keyword: str, mask_text: str="[MASK]"):
    return text.replace(keyword, mask_text)

print(mask(text, textrank.keywords()[0]))

class QuizGenerator(object):
    def __init__(self, text):
        self.text = text
        self.textrank = TextRank(self.text)
        self.summarize = self.textrank.summarize
        self.keywords = self.textrank.keywords()

    def mask(self, keyword: str, mask_text: str="[MASK]"):
        masked = self.text.replace(keyword, mask_text)
        return masked

    def generate_quiz(self):  # Generates quiz with given string. 
        keyword = self.keywords[0]  # Temporal
        return self.mask(keyword), keyword  # problem, answer

quizgenerator = QuizGenerator(text)
problem, answer = quizgenerator.generate_quiz()

print(problem)
print(answer)

text = """
오늘날 민주주의의 내용은 복잡하고 다의성을 띠지만, 민주주의 국가가 되려면 최소한 ‘1) 국민의 기본권 존중, 2) 권력의 전제화를 억제할 여러 중요한 정치제도 확립’이 충족되어야 한다. 이 조건 두 개가 충족되지 못한 국가는 어떠한 뜻에서도 민주주의 국가가 아니다. 근대 민주주의의 역사는 이 두 가지 조건을 확립하고 발전시킨 것이다.

그렇다면 국민의 기본권이 무엇인지가 의문으로 떠오른다.  정치과학자 레리 다이아몬드 (Larry Diamond)는  민주주의 사회의 기본권에 대해서 다음과 같이 설명했다. 1)  선택권 - 자유롭고 공정한 선거를 통해 정부를 선택하고 교체할 수 있는 정치 제도에 국민들이 참여할 수 있고, 2) 참정권 - 정치 및 시민 생활에 국민들이 적극적으로 참여할 수 있으며, 3) 인권 - 모든 국민의 인권이 보장되며, 마지막으로 4) 평등법칙 - 모든 국민에게 법률 및 절차가 동등하게 적용되는 법치 사회가 보장되어야 한다고 했다. 여기서 우리가 주목할 것은 진정한 민주주의 국가에서는 불법 판결을 받은 사람에게도 인권과 평등법칙은 언제나 누릴 수 있는 기본권이라는 것이다. 하물며 국민이 불법 판결을 받기 전에 그 사람의 인권이 유린된다면 그 사회는 민주주의라 볼 수 없다.

그러나 다수의 의견을 따르는 것이 모두 민주주의에 부합하는지에 대해서는 오랫동안 논란이 제기돼왔다. 허정은 절대다수의 주장이 곧 민주주의라는 견해에 홀로 반대하기도 했다. 허정에 의하면 '사람의 머리수, 정당 당원들의 총 수가 많다는 것이 정당의 우수성의 증명은 아니다.'라고 주장하기도 했다. 허정은 소수의 의견이라고 해도 합리적이고 올바른 주장이면 수용하는 것이 민주주의라 주장했다.

허정의 이러한 우려는 순수 민주주의의 약점이다. 그리고 이렇게 모든 것을 절대다수의 결정을 따르는 순수 민주주의의 약점을 보강한 것이 공화제이다. 사실 지금도 다수를 존중하되 소수를 배려하는 것이 민주주의 국가의 기본 모토이긴 하다. 하지만 현실에선 사회적 소수자들이 탄압받을 가능성도 있기 때문에 주의할 부분. 아나키즘도 이런 관점에서 생겨난 측면이 있다.

한편 프리드리히는 선거제도를 '헌법국가의 기초적인 결단사항'으로 언급한 바 있다.
"""

quizgenerator = QuizGenerator(text)
problem, answer = quizgenerator.generate_quiz()
print(problem)
print()
print(answer)