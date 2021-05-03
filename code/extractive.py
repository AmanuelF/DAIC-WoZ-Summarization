from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from data_prep import get_simple_passages, get_data, to_passage_simple, get_summaries


def main_pruned(n_sentences=4, fraction=None):
    passages = get_simple_passages()
    summarizers = {}
    for pid, text in passages.items():
        if fraction is not None:
            n_sentences = int(fraction * text.count('.'))
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizers['LexRank'] = LexRankSummarizer()
        summarizers['Luhn'] = LuhnSummarizer()
        summarizers['Lsa'] = LsaSummarizer()
        summarizers['TextRank'] = TextRankSummarizer()
        summarizers['SumBasic'] = SumBasicSummarizer()
        summarizers['KL'] = KLSummarizer()
        for method, summarizer in summarizers.items():
            summary = summarizer(parser.document, n_sentences)
            # print(len(summary))
            with open(f'../extractive_summaries/{method}/{pid}.txt', 'w') as fp:
                for sentence in summary:
                    fp.write(str(sentence) + '\n')


def main_unpruned(foldername, n_sentences=4, fraction=None, same_n_abs=False):
    data, _ = get_data()
    passages = {}
    for pid, convo in data.items():
        passages[pid] = to_passage_simple(convo)
    if same_n_abs:
        abs_summaries = get_summaries(augmented_lm=False)
    for pid, text in passages.items():
        if same_n_abs:
            n_sentences = abs_summaries[pid].count('.')
        elif fraction is not None:
            n_sentences = int(fraction * text.count('.'))
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = SumBasicSummarizer()
        summary = summarizer(parser.document, n_sentences)
        # print(len(summary))
        with open(f'../extractive_summaries/{foldername}/{pid}.txt', 'w') as fp:
            for sentence in summary:
                fp.write(str(sentence) + '\n')


if __name__ == '__main__':
    # main_unpruned('SumBasic_unpruned', same_n_abs=True)
    main_unpruned('SumBasic_unpruned_more', fraction=0.4)
