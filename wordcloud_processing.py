# Import the wordcloud library
from wordcloud import WordCloud
import pandas as pd
# from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords
import random
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import markdown as md

from sklearn.feature_extraction.text import TfidfVectorizer

from stopwords import stopwords


def pink_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
	return "hsl(327, 100%%, %d%%)" % random.randint(25, 80)

def pink_color_func_constant(word, font_size, position, orientation, random_state=None, **kwargs):
	return "hsl(330, 100%, 79%)"

def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
	return("hsl(0,100%, 1%)")

def strip_char(w):
	for ii in range(6):
		w = w.strip(' ')
		w = w.strip('.')
		w = w.strip(',')
		w = w.strip('-')
		w = w.strip(':')
		w = w.strip('(')
		w = w.strip(')')
		w = w.strip(';')
		# w = w.strip('xb')
		w = w.strip('\'')
	return w

def clean_corpus(text_path):
    # papers = pd.read_pickle(text_path)
    if isinstance(text_path, str):
        with open(text_path, "r") as istr:
            data = [i.strip() for i in istr.readlines()]
        papers = pd.DataFrame([{"id": i, "description": t} for i, t in enumerate(data)])
    else:
        papers = pd.DataFrame(text_path)

    
    

    to_exclude = ['page', 'table', 'http', 'et al', 'conclusion', 'm', 't', 'c', 'et', 'al', 'data', 'acknowledgements', 'publisher', 'Authors', 'details', 'contribution', 'Author', 'result',
                    'results', 'study', 'found conclusion', 'using', 'system', 'method', 'sample', 'value', 'project', 'thus', 'increase', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                    'may', 'parameter', 'due', 'given', 'decrease', 'first', 'low', 'lower', 'e  g', 'high', 'case', 'condition', 'information', 'calculation', 'n.a', 'value', 'change', 'sample', 'show', 'work', 'license', 'article', 'ystem',
                    'etween', 'remains', 'neutral', 'springer', 'http', 'www', 'www ', 'manuscript', 'authors', 'gmbh', 'papers', 'analysi', 'author', 'acknowledgement', 'http', 'common', 'upper', 'small', 'number', 'non', 'remain', '000', '10 ', 'studie',
                    'regard', 'final', 'credit', 'paper', 'thesi', 'tion', 'would', 'relationship', 'ass', 'western', 'without', 'working', 'claim', 'detail', 'abstract', 'found', 'large', 'could', 'keyword', 'north', 'medium', 'location', 'acces',
                    'general']
    list_words = []
    to_keep = ['stress', 'properties', 'process']

    stop_words = set(stopwords)
    all_sentences = papers['description'].tolist()

    for i, s in enumerate(all_sentences):
        sentence = s.split()
        new_sentence = []
        for w in sentence:
            w = strip_char(w.lower())
            letters = [char for char in w]
            to_add = False
            if len(letters) > 4:
                try:
                    u = float(w)
                    to_add = False
                except:
                    to_add = True
                    pos = 'n'
                # if len(wn.synsets(w)) > 0: pos = wn.synsets(w)[0].pos()
                if w == 'analyses': w = 'analysis'
                if w == 'technologie': w = 'technology'
                if (w not in to_keep): w = w[:-1]
                if (w in stop_words) or (pos != 'n') or (w.endswith('ed')) or (w in to_exclude): to_add = False
                if 'http' in w: to_add = False
                if '0' in w: to_add = False
                if '1' in w: to_add = False
                if '(' in w: to_add = False
                if '\'' in w: to_add = False
            if to_add:
                list_words.append(w)
                new_sentence.append(w)
        # print(i, ' '.join(new_sentence))
        # print(i)
        papers.iloc[[i],[1]] = ' '.join(new_sentence)

    return papers, list_words

def get_tf_idf(corpus):

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.001, max_df = 0.6, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    # print(df)
    return df, feature_names

def compute_wc(df, feature_names):
    frequencies = []
    for c in df.columns:
        summation = np.sum(df[c])
        frequencies.append(summation)
    frequencies = np.array(frequencies)
    dff = pd.DataFrame(frequencies.reshape((1,frequencies.shape[0])), columns=feature_names)
    data = dff.transpose()
    data.columns = ['word_list']

    # wordcloud = WordCloud(font_path = '/Library/Fonts/Arial Unicode.ttf', background_color="#f0f0f0", width=1920, height=1080, max_words=500).generate_from_frequencies(data['word_list'])
    wordcloud = WordCloud(background_color="#f0f0f0", width=1920, height=1080, max_words=500).generate_from_frequencies(data['word_list'])
    black = io.BytesIO()
    fig = plt.figure(figsize = (15,10))
    plt.imshow(wordcloud.recolor(color_func=black_color_func, random_state=3), interpolation="bilinear")
    plt.axis("off")
    fig.savefig(black, format='png', bbox_inches="tight", transparent = True)
    plt.close(fig)
    black = base64.b64encode(black.getvalue()).decode("utf-8").replace("\n", "")

    # wordcloud = WordCloud(font_path = '/Library/Fonts/Arial Unicode.ttf', background_color="#0f0628", width=1920, height=1080, max_words=500).generate_from_frequencies(data['word_list'])
    wordcloud = WordCloud(background_color="#0f0628", width=1920, height=1080, max_words=500).generate_from_frequencies(data['word_list'])
    pink = io.BytesIO()
    fig = plt.figure(figsize = (15,10))
    plt.imshow(wordcloud.recolor(color_func=pink_color_func_constant, random_state=3), interpolation="bilinear")
    plt.axis("off")
    fig.savefig(pink, format='png', bbox_inches="tight", transparent = True)
    plt.close(fig)
    pink = base64.b64encode(pink.getvalue()).decode("utf-8").replace("\n", "")
    
    return black, pink

class HTMLDoc:
    def __init__(self):
        self.markdown = ''
        self.html = None

    def add_text(self, text):
        self.markdown += text
        self.markdown += '\n'

    def add_bytestring_image(self, bytestring, alt_text = 'alt_text'):
        image_string = '![' + alt_text + '](data:image/png;base64,' + bytestring + ')'
        self.markdown += image_string
        self.markdown += '\n'

    def add_image(self, image, alt_text = 'alt_text'):
        image_string = '![' + alt_text + '](' + image + ')'
        self.markdown += image_string
        self.markdown += '\n'

    def add_css(self, css_file):
        css_string = '<html>\n<head>\n<link rel=\"stylesheet\" href=\"' + css_file + '\">\n</head>'
        self.html = css_string + self.html
        self.html += '</html>'

    def to_html(self):
        self.html = md.markdown(self.markdown)

def prepare_html_output(s1, s2):
    text = '''# Word Cloud

        '''

    style = '''<html><head>
            <style>
                * {
                font-family: 'Roboto', sans-serif;
                }
                h1 {
                text-align: center;
                }
                p {
                text-align: left;
                }
            </style>
            </head>
            '''

    doc = HTMLDoc()
    doc.add_text(text)
    doc.add_bytestring_image(s1)
    doc.add_bytestring_image(s2)
    doc.to_html()
    doc.html = style + doc.html
    doc.html += '</html'
    # doc.add_css("styling.css")

    return doc.html