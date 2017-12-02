#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import fasttext
from SentenceSplit import *

class TrainModel:
	__traintext = "train.txt"
	__outputtext = "trainfile.txt"
	__spliter = SentenceSplit()
	__classifier = None
	
	def __init__ (self, inputdir, model):
		self.inputdir = inputdir
		self.model = model
		self.__foutputtext = None
	
	def __concatText (self,traindir,label):
		filepath = os.path.join(traindir,TrainModel.__traintext)
		with open(filepath) as f:
			for text in f :
				text = text.replace('\n','').replace('\r','')
				text_seg = self.__spliter.splitSentence(text).replace('\n','')
				text_seg = text_seg + '\t__label__' + label + '\n'
				self.__foutputtext.write(text_seg)
		self.__foutputtext.flush()
	
	def __getTrainText (self,textdir):
		lists = os.listdir(textdir)
		for list in lists:
			abspath = os.path.join(textdir,list)
			if os.path.isdir(abspath):
				label = list
				self.__concatText(abspath,label)
				self.__getTrainText(abspath)
				
	def __closeoutputtext (self):
		if not ( self.__foutputtext is None and self.__foutputtext.closed):
			self.__foutputtext.close()
	
	def __openoutputtext (self):
		if self.__foutputtext is None:
			foutputpath = os.path.join(self.inputdir,self.__outputtext)
			self.__foutputtext = open (foutputpath,"w")
	
	def getClassifier (self):
		modelname = self.model + '.bin'
		modelpath = os.path.join(self.inputdir,modelname)
		if TrainModel.__classifier is None:
			if not os.path.isfile(modelpath):
				self.__openoutputtext()
				self.__getTrainText(self.inputdir)
				self.__closeoutputtext()
				outputtextpath = os.path.join(self.inputdir,self.__outputtext)
				TrainModel.__classifier = fasttext.supervised(outputtextpath,self.model)
				oscommand = 'mv ' + modelname + ' ' + self.inputdir
				os.system(oscommand)
				return TrainModel.__classifier
			else:
				TrainModel.__classifier = fasttext.load_model(modelpath)
				return TrainModel.__classifier
		else:
			return TrainModel.__classifier
