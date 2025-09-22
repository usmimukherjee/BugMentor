# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import re
import string

from nltk import word_tokenize
from utils.str_util import unicode2str
import six


# question unit
def preprocessing_for_que(q):
    q.title = preprocess_title(q.title)
    q.body = preprocess_body(q.body)
    q.tag = preprocessing_for_tag(q.tag)
    return q


def preprocessing_for_tag(tag_str):
    return tag_str.replace('<', ' ').replace('>', ' ').strip().split()


def remove_punctuation(input_string):
    """
    Removes all punctuation characters from a given input string.
    """
    return ''.join(char for char in input_string if char not in string.punctuation)



def preprocess_title(title):
    text = title.lower()
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    text = remove_punctuation(text)
    return text.strip()


def preprocess_body(body):
    text = body.lower()
    text = remove_text_code(text)
    text = remove_html_tags(text)
    # text = str(text)[2:-1]
    text = re.sub(r'[\ud800-\udfff]', '', text)
    #join list to string
    # text = ' '.join(text)
    # text = text.encode('utf-8')
    #remove byte mark from string
    # text = unicode2str(text)
    # print(type(text))
    #replace new line with space
    # text = text.replace('\n', ' ')
    text = re.sub(r'\n', ' ', text)

    text = replace_double_space(text)
    text = tokenize_and_rebuild(text)
    text = remove_punctuation(text)
    if text:
        return text.strip()
    return ''


def remove_text_code(html_str):
    import re
    # regex: <pre(.*)><code>([\s\S]*?)</code></pre>
    regex_pattern = r'<pre(.*?)><code>([\s\S]*?)</code></pre>'
    html_text = html_str
    for m in re.finditer(regex_pattern, html_str):
        raw_code = html_str[m.start():m.end()]
        # remove code
        html_text = html_text.replace(raw_code, " ")
    return html_text.replace('\n', ' ')


# answer unit
def preprocessing_for_ans(ans):
    text = remove_text_code(ans.body.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    ans.body = text.strip()
    return ans


def preprocessing_for_ans_sent(sent):
    text = remove_text_code(sent.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    return text.strip()


def replace_double_space(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


def remove_html_tags(raw_html):
    from bs4 import BeautifulSoup
    try:
        text = BeautifulSoup(raw_html, "html.parser").text
    except Exception as e:
        # UnboundLocalError
        text = clean_html_tags2(raw_html)
    finally:
        if isinstance(text, bytes):
            return text.encode('utf-8')
        else:
            return text



def clean_html_tags2(raw_html):
    import re
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def tokenize_and_rebuild(sent):
    # sent = six.text_type(sent, 'utf-8')
    # sent = str(sent, 'utf-8')
    sent = sent.encode().decode('utf-8')
    ws = word_tokenize(sent)
    return ' '.join(ws)
    # return unicode2str(' '.join(ws))


if __name__ == '__main__':
    text = 'text > <img src= http://i.stack.imgur.com/ltCod.    png alt= Rich task editor >   '
    print(remove_html_tags(text))
    print(replace_double_space(text))
