import re
from bert_dp.tokenization import FullTokenizer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path


@register('syntax_parser_preprocessor')
class SyntaxPreprocessor(Component):
    def __init__(self, vocab_file, syntax_parser, **kwargs):
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.syntax_parser = syntax_parser
        
    def __call__(self, texts):
        good_texts = []
        good_text_numbers = []
        bs = len(texts)
        for i in range(bs):
            text_tokens = re.findall(self.re_tokenizer, texts[i])
            text_len = sum([len(self.tokenizer.tokenize(token)) for token in text_tokens])
            if len(texts[i]) > 1 and text_len < 500:
                good_texts.append(texts[i])
                good_text_numbers.append(i)
        
        processed_texts = ["" for _ in texts]
        if good_texts:
            syntax_parser_res_batch = self.syntax_parser(good_texts)
            for text_num, syntax_parser_res in zip(good_text_numbers, syntax_parser_res_batch):
                processed_texts[text_num] = syntax_parser_res
        
        return processed_texts
