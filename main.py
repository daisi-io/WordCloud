from datetime import datetime
import pandas as pd

from wordcloud_processing import clean_corpus, get_tf_idf, compute_wc, prepare_html_output


def compute(text_path:pd.DataFrame):
    start = datetime.now()
    papers, list_words = clean_corpus(text_path)
    print(f"finish cleaning text: {datetime.now() - start}")

    start = datetime.now()
    df, feature_names = get_tf_idf(papers['description'])
    black, pink = compute_wc(df, feature_names)
    print(f"finish compute word cloud: {datetime.now() - start}")
    
    html_out = prepare_html_output(black, pink)

    return [{"type": "html", "data": html_out}]

# def schema():
#     r = [
#         {
#             "id":    "text_path",
#             "type":  "file",
#             "label": "paragraph text path",
#             "props": {}
#         }
#     ]

#     return r

# if __name__ == "__main__":
#     path = '/Users/zhenshanjin/Documents/Belmont/sandy/UtilityDaisies/WordCloud/abstracts.p'
#     res = compute(path)