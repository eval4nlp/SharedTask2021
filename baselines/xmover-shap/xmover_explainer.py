import sys
sys.path.append('./xmover')
import pandas as pd
import numpy as np
import shap
import os
from mosestokenizer import MosesDetokenizer, MosesTokenizer
import argparse
from scorer import XMOVERScorer
import numpy as np
import torch
import truecase


class XMoverWrapper():
    def __init__(self, src_lang, tgt_lang, model_name, do_lower_case, language_model, mapping, device, ngram, bs):
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang
        self.mapping = mapping
        self.device = device
        self.ngram = ngram
        self.bs = bs

        temp = np.loadtxt('./xmover/mapping/europarl-v7.' + src_lang + '-' + tgt_lang + '.2k.12.BAM.map')
        self.projection = torch.tensor(temp, dtype=torch.float).to(device)
        
        temp = np.loadtxt('./xmover/mapping/europarl-v7.' + src_lang + '-' + tgt_lang + '.2k.12.GBDD.map')
        self.bias = torch.tensor(temp, dtype=torch.float).to(device)

        self.scorer = XMOVERScorer(model_name, language_model, do_lower_case, device)

        self.src_sent = None

    def __call__(self, translations):
        assert self.src_sent is not None

        translations = [s[0] for s in translations]
        translations = [truecase.get_true_case(s) for s in translations]
        source = [self.src_sent] * len(translations)
        xmoverscores = self.scorer.compute_xmoverscore(self.mapping, self.projection, self.bias, source, translations, self.ngram, self.bs)
        return np.array(xmoverscores)

    def tokenize_sent(self, sentence, lang):
        # with MosesTokenizer(lang) as tokenize:        
            # tokens = tokenize(sentence)
        # return tokens       
        return sentence.split()

    def detokenize(self, tokens, lang):
        # with MosesDetokenizer(lang) as detokenize:        
            # sent = detokenize(tokens)
        # return sent 
        return ' '.join(tokens)

    def build_feature(self, trans_sent):
        tokens = self.tokenize_sent(trans_sent, self.tgt_lang)
        tdict = {}
        for i,tt in enumerate(tokens):
            tdict['{}_{}'.format(tt,i)] = tt

        df = pd.DataFrame(tdict, index=[0])
        return df

    def mask_model(self, mask, x):
        tokens = []
        for mm, tt in zip(mask, x):
            if mm: tokens.append(tt)
            else: tokens.append('[MASK]')
        trans_sent = self.detokenize(tokens, self.tgt_lang)
        sentence = pd.DataFrame([trans_sent])
        return sentence



class ExplainableXMover():
    def __init__(self, src_lang, tgt_lang, model_name='bert-base-multilingual-cased', do_lower_case=False, language_model='gpt2', mapping='CLP', device='cuda:0', ngram=2, bs=32):
        self.wrapper = XMoverWrapper(src_lang, tgt_lang, model_name, do_lower_case, language_model, mapping, device, ngram, bs)
        self.explainer = shap.Explainer(self.wrapper, self.wrapper.mask_model)

    def __call__(self, src_sent, trans_sent):
        #return self.wrapper.scorer.compute_xmoverscore(self.wrapper.mapping, self.wrapper.projection, self.wrapper.bias, [self.wrapper.detokenize(src_sent.split(),self.wrapper.src_lang)], [self.wrapper.detokenize(trans_sent.split(),self.wrapper.tgt_lang)], self.wrapper.ngram, self.wrapper.bs)[0]
        self.wrapper.src_sent = src_sent
        return self.wrapper([[trans_sent]])[0]


    def explain(self, src_sent, trans_sent, plot=False):
        self.wrapper.src_sent = src_sent
        value = self.explainer(self.wrapper.build_feature(trans_sent))
        if plot: shap.waterfall_plot(value[0])
        all_tokens = self.wrapper.tokenize_sent(trans_sent, self.wrapper.tgt_lang)

        return [(token,sv) for token, sv in zip(all_tokens,value[0].values)]


if __name__ == '__main__':
    model = ExplainableXMover('et','en')
    src = 'Kant ütleb , et kõik käsitööd , tööndused ja kunstid on tööjaotusest võitnud ning filosoofia peaks neist eeskuju võtma .'
    trans = 'Kant says that all crafts , all works and all the arts have gained from the division of labour , and the philosophy should take inspiration from them .'
    score = model(src, trans)
    exps = model.explain(src, trans)
    
    print('\n =========')
    print(score)
    print(exps)




