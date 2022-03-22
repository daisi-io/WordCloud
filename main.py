from datetime import datetime
import pandas as pd

from wordcloud_processing import clean_corpus, get_tf_idf, compute_wc, prepare_html_output


def generate_wordcloud(texts):
    # print(f"input path: {texts_path}")
    start = datetime.now()
    text_df = pd.DataFrame(texts)
    papers, list_words = clean_corpus(text_df)
    print(f"finish cleaning text: {datetime.now() - start}")

    start = datetime.now()
    df, feature_names = get_tf_idf(papers['description'])
    black, pink = compute_wc(df, feature_names)
    print(f"finish compute word cloud: {datetime.now() - start}")
    
    html_out = prepare_html_output(black, pink)

    return [
        {"type": "html", "data": html_out},
        {"type": "image", "label": "black", "data":  {"alt": "Black WordCloud", "src": "data:image/png;base64, " + black}},
        {"type": "image", "label": "pink", "data":  {"alt": "Pink WordCloud", "src": "data:image/png;base64, " + pink}},
    ]
