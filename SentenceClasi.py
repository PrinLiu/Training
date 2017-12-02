#!/usr/bin/python
# -*- coding: UTF-8 -*-

from TrainModel import *
from SentenceSplit import *
import fasttext

class SentenceClasi:
	__spliter = SentenceSplit()
	
	def __init__ (self, classifier):
		self.classifier = classifier
	
	def classifySentence (self, sentence):
		#sent = self.__spliter.splitSentence(sentence)
		label = self.classifier.predict(sentence)
		return label
		
	def classPrecision (self, sentence):
		#sent = self.__spliter.splitSentence(sentence)
		result = self.classifier.test(sentence)
		return result

		
if __name__ == '__main__' :
	model1 = TrainModel("/home/prin/fttrain","train")
	sspliter = SentenceSplit()
	clasifier = model1.getClassifier()
	sentclasi = SentenceClasi(clasifier)
	str1 = u"早，汇丰银行，我想查一下我账户"
	str1 = sspliter.splitSentence(str1)
	str1 = str(str1)
	texts = [str1]
	print (texts)
	label = sentclasi.classifySentence(texts)
	#result = sentclasi.classPrecision(texts)
	print (label)
	 #打印结果:[['check']]
	#print (result.precision)
	#print (result.recall)
	str2 = u"HSBC，帮我转帐"
	str2 = sspliter.splitSentence(str2)
	str2 = str(str2)
	texts = [str2]
	print (texts)
	label = sentclasi.classifySentence(texts)
	print (label)
	 #打印结果:[[]'forward']
