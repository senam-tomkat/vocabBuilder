import sys
import re
import itertools
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file = sys.argv[2]

    magoosh_word_list = pd.read_csv("magoosh_word_list.csv")
    manhat_word_list = pd.read_csv("manhattan_word_list.csv")

    word_list = magoosh_word_list.merge(manhat_word_list, how='outer', on=['word'], suffixes=('', '_manhat'))
    word_list["definition"] = word_list["definition"].where(word_list["definition"].notnull(),
                                                            word_list["definition_manhat"])
    word_list = word_list.drop_duplicates(subset=["word"], keep='first')
    word_list = word_list[["word", "definition"]]

    sb_stemmer = SnowballStemmer("english")

    with open(input_file_path, 'r') as doc:
        text = doc.read()
        sent_list = sent_tokenize(text)
        common_word_df_list = []
        for sent_0, sent_1 in pairwise(sent_list):
            word_sent_stem_set = set(map(sb_stemmer.stem, word_tokenize(sent_1)))
            word_sent_raw_set = set(word_tokenize(sent_1))
            word_sent_full_set = word_sent_stem_set.union(word_sent_raw_set)
            word_sent_df = pd.DataFrame(word_sent_full_set, columns=["word"])
            df_common_words = word_list.merge(word_sent_df, how="inner", on=["word"])
            if not df_common_words.empty:
                context_sentence = sent_0 + " " + sent_1    # concatinate sentence before the matched sentence
                df_common_words["sentence"] = context_sentence
                # df_common_words["sentence"] = sent_1
                common_word_df_list.append(df_common_words)

        df_common_word_final = pd.concat(common_word_df_list)
        df_common_word_final["sentence"] = df_common_word_final["sentence"].str.replace(pat=r"\(cid:[0-9]*?\)",
                                                                                        repl="", regex=True)
        df_common_word_final["sentence"] = df_common_word_final["sentence"].map(lambda x: re.sub(r'\n+', ' ', x))
        df_common_word_final = df_common_word_final.drop_duplicates(subset=["word"], keep='first')
        df_common_word_final = df_common_word_final[["word", "definition", "sentence"]]

        css_style = """
        <style>
            .word {
            }
            .def {
                font-size: 14px;
                font-weight: bold;
            }
            .context {
                font-size: 13px;
            }
            .word-div {
                margin: 10px 0;
            }
        </style>
        """
        base_html = """
        <html>
            <head>
                {style}
            </head>
            <body>{body}</body>
        </html>"""
        body = ""
        for index, row in df_common_word_final.iterrows():
            pat = r"({}.*?)([, ;:!\.])"
            sentence = re.sub(pat.format(row["word"]), r"<b><i>\1</i></b>\2".format(row["word"]), row["sentence"])
            body += """
            <div class="word-div">
                <span class="word"><b>{word}</b></span>
                <br>
                <span class="def"><i>def: </i>{definition}</b></span>
                <br>
                <span class="context">Usage: {sentence}</span>
            </div>
            """.format(word=row["word"], definition=row["definition"], sentence=sentence)

        html_str = base_html.format(style=css_style, body=body)

        with open(output_file, "w") as f:
            f.write(html_str)
