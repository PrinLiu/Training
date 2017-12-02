#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import os

class SentenceSplit:
	
	def __init__ (self):
		pass
	
	def splitSentence (self,sentence):
      '''
        Split the input sentence and return with a joined string
      '''
		result = jieba.cut(sentence)
		result = list(result)
		result = " ".join(result)
		return result
