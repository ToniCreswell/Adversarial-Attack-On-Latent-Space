#VAE trained on celebA faces
import sys
sys.path.append('../')

from dataload import CELEBA
from function import make_new_folder, plot_losses, vae_loss_fn, save_input_args,\
 is_ready_to_stop_pretraining, plot_norm_losses
from function import binary_class_score as score
from models import VAE
from models import DISCRIMINATOR as CLASSIFIER

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import os
from os.path import join

import argparse

from PIL import Image

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from time import time

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data/datasets/LabelSwap', type=str)
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--maxEpochs', default=10, type=int)
	parser.add_argument('--nz', default=100, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	parser.add_argument('--outDir', default='../../Experiments_delta_z/celeba_joint_VAE_DZ/', type=str)
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--loadVAE', action='store_true')
	parser.add_argument('--load_VAE_from', type=str)
	parser.add_argument('--loadCLASSER', action='store_true')
	parser.add_argument('--load_CLASSER_from', default='../../Experiments_delta_z/celeba_joint_VAE_DZ/Ex_15', type=str)
	parser.add_argument('--loadDELTAZ', action='store_true')
	parser.add_argument('--load_DELTAZ_from', type=str)
	parser.add_argument('--jointClassLoss', action='store_true')
	parser.add_argument('--sig', default=1, type=float)  #std of the prior
	parser.add_argument('--rho', default=1.0, type=float)  # weight on the regulariation term
	parser.add_argument('--alpha', default=1, type=float)  # weight on the KL loss in the VAE loss
	parser.add_argument('--phi', default=1, type=float)  #weight on the class loss on reconstructed samples
	parser.add_argument('--evalMode', action='store_true')
	parser.add_argument('--p', default=1, type=int)
	return parser.parse_args()


def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size(0),1).type_as(x)
	return x,y

def viz_DZ(DZ, exDir):
	fig1 = plt.figure()
	plt.stem(DZ)
	plt.xlabel('z component')
	plt.ylabel('value')
	plt.savefig(join(exDir, 'deltaZ.png'))

def eval_results(vae, classer, deltaZ, testLoader, exDir):
	#Evaluate the results:
	vae.eval()
	classer.eval()
	# Take a test set of smiling and non-smiling people
	testData = iter(testLoader).next()
	x, y = prep_data(testData, classer.useCUDA)
	
	smileIdx = torch.nonzero(y.data)
	noSmileIdx = torch.nonzero(1 - y.data)

	xSmile = torch.index_select(x, dim=0, index=smileIdx[:,0])
	xNoSmile = torch.index_select(x, dim=0, index=noSmileIdx[:,0])

	images = {'smile': xSmile, 'no_smile': xNoSmile }

	# for a set of {similar and not smiling}
	for label in images:
		print 'processing the', label, 'images'
		x = images[label]
		#Visualise them
		save_image(x.data, join(exDir, label+'_original.png'))

		#reconstruct, visualise
		recX, mu, logVar = vae.forward(x)
		save_image(recX.data, join(exDir, label+'_rec.png'))

		#Encode them, +/- deltaZ, decode them, visualise
		z = vae.re_param(mu, logVar)
		if label is 'smile':
			zSwap = z + deltaZ
		else:
			zSwap = z - deltaZ

		xSwap = vae.decode(zSwap)
		save_image(xSwap.data, join(exDir, label+'_swap.png'))

		#save a difference images
		save_image(torch.abs(xSwap.data - recX.data), join(exDir, label+'_DX.png'))

def inception_score(vae, classer, deltaZ, testLoader, exDir):
	'print calc inception score..'
	vae.eval()
	classer.eval()
	# Take a test set of smiling and non-smiling people
	numSmile=0
	numNoSmile=0
	# for a set of {similar and not smiling}
	scores = {'smileO':0, 'smileR':0, 'smileSW':0, 'no_smileO':0, 'no_smileR':0, 'no_smileSW':0}
	for i, testData in enumerate(testLoader):
		x, y = prep_data(testData, classer.useCUDA)
	
		smileIdx = torch.nonzero(y.data)
		noSmileIdx = torch.nonzero(1 - y.data)

		numSmile+=len(smileIdx)
		numNoSmile+=len(noSmileIdx)

		print 'num smile:', numSmile
		print 'num no smile:', numNoSmile

		xSmile = torch.index_select(x, dim=0, index=smileIdx[:,0])
		xNoSmile = torch.index_select(x, dim=0, index=noSmileIdx[:,0])

		images = {'smile': xSmile, 'no_smile': xNoSmile }

		# 'O' original, 'R', reconstruction, 'SW', swapped
		for label in images:
			print 'processing the', label, 'images'
			#original images and score
			x = images[label]
			conf = classer.forward(x)
			if label is 'no_smile':
				conf = 1 - conf
			scores[label+'O']+=(conf).sum().data[0]


			#reconstruct and score
			recX, mu, logVar = vae.forward(x)
			conf = classer.forward(recX)
			if label is 'no_smile':
				conf = 1 - conf
			scores[label+'R']+=(conf).sum().data[0]

			#Encode them, +/- deltaZ, decode them, visualise
			z = vae.re_param(mu, logVar)
			if label is 'smile':
				zSwap = z + deltaZ
			else:
				zSwap = z - deltaZ

			xSwap = vae.decode(zSwap)
			conf = classer.forward(xSwap)
			if label is 'smile':  #cause label switch
				conf = 1 - conf

			scores[label+'SW']+=(conf).sum().data[0]

	#divide scores by i+1 and save them in a .txt file
	f = open(join(exDir, 'scores.txt'), 'w')
	f.write('High confidence = 1, low confidence = 0 \n')
	f.write('Original smile:' + str(scores['smileO']/numSmile)+'\n')
	f.write('Smile reconstruction:' + str(scores['smileR']/numSmile)+'\n')
	f.write('Smile label switch:' + str(scores['smileSW']/numSmile)+'\n')
	f.write('Original no smile:' + str(scores['no_smileO']/numNoSmile)+'\n')
	f.write('No smile reconstruction:' + str(scores['no_smileR']/numNoSmile)+'\n')
	f.write('No smile label switch:' + str(scores['no_smileSW']/numNoSmile)+'\n')
	f.write('commit:'+opts.commit)
	f.close()


if __name__=='__main__':
	opts = get_args()

	####### Data set #######
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
	trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor())
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor())
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)


	####### Create VAE model and classifier model #######
	vae = VAE(nz=opts.nz, imSize=64, fSize=opts.fSize, sig=opts.sig)
	classer = CLASSIFIER(imSize=64, fSize=opts.fSize)
	print vae
	print classer

	if vae.useCUDA:
		print 'using CUDA'
		vae.cuda()
		classer.cuda()
	else: print '\n *** NOT USING CUDA ***\n'

	#eval or train
	if opts.evalMode:
		opts.loadCLASSER = True
		opts.loadDELTAZ = True
		opts.loadVAE = True
	else:
		#create a folder to save the exp
		exDir = make_new_folder(opts.outDir) # Create a new folder to save results and model info
		print 'Outputs will be saved to:',exDir
		save_input_args(exDir, opts)

	####### Load pre-trained model or train a model #######
	#Load pre-trained nets if available
	if opts.loadVAE:
		print 'loadding vae...'
		vae.load_params(opts.load_VAE_from)
	if opts.loadCLASSER:
		print 'loadding classer...'
		classer.load_params(opts.load_CLASSER_from)
	if opts.loadDELTAZ:
		print 'loading deltaZ...'
		deltaZ = torch.load(join(opts.load_DELTAZ_from, 'deltaZ'))
	else:
		if vae.useCUDA:
			deltaZ = Variable(torch.randn(1,opts.nz).cuda(), requires_grad=True)
		else:
			deltaZ = Variable(torch.randn(1,opts.nz), requires_grad=True)

	if opts.evalMode:
		inception_score(vae, classer, deltaZ, testLoader, opts.load_VAE_from)
		eval_results(vae, classer, deltaZ, testLoader, opts.load_VAE_from)
		exit()

	####### Define optimizers for each set of params #######
	optimizer_VAE = optim.Adam(vae.parameters(), lr=opts.lr)  #specify the params that are being upated
	optimizer_CLASSER = optim.Adam(classer.parameters(), lr=opts.lr)
	optimizer_DZ = optim.Adam([deltaZ], lr=opts.lr)

	vaeLosses = {'kl':[], 'bce':[], 'test_bce':[], 'test_kl':[]}
	classerLosses = {'bce':[], 'train_acc':[], 'test_acc':[]}
	deltaZLosses = {'total':[], 'reg':[]}
	totalLosses = {'total':[]}
	Nb = len(trainLoader) #no of batches
	####### Start Training #######
	for e in range(opts.maxEpochs):
		vae.train()
		classer.train()

		TIME = time()

		for i, data in enumerate(trainLoader, 0):

			x,y = prep_data(data, useCUDA=vae.useCUDA)

			#zero the grads - otherwise they will be acculated
			#Done below now
	

			####### VAE #######
			#get ouput, clac loss, calc all grads, optimise
			optimizer_VAE.zero_grad()
			outRec, outMu, outLogVar = vae(x)
			bceLoss, klLoss = vae.loss(rec_x=outRec, x=x, mu=outMu, logVar=outLogVar)
			loss = bceLoss + opts.alpha*klLoss 
			loss.backward(retain_graph=True) #fill in grads
			optimizer_VAE.step()
			#DO the optimization step later - cause using a reconstruction loss to do a step too

			####### CLASSER #######
			#get ouput, clac loss, calc all grads, optimise

			## - 3 components to the classification loss
			# #1 classification loss on the training data smaples
			# #2 classifcation loss on the reconstructed data samples
			# #3 classification loss on the flipped samples - DO NOT USE TO UPDATE CLASSIFIER - USED TO UPDATE DELTA Z!
			optimizer_CLASSER.zero_grad()
			predY = classer.forward(x)
			classLoss = bce(predY.type_as(y), y)
			classLoss.backward(retain_graph=True)
			optimizer_CLASSER.step()

			optimizer_VAE.zero_grad()
			if opts.jointClassLoss:
				predYrec = classer.forward(outRec) #Do not update classer with this loss!
				classLossRec = opts.phi * bce(predYrec.type_as(y), y)
				classLossRec.backward(retain_graph=True)  #will be updating the encoder and decoder!!! can detach else where to NOT do this!
			optimizer_VAE.step()


			####### DELTA Z #######
			optimizer_DZ.zero_grad()
			z = vae.re_param(outMu, outLogVar)
			zSwap = z + torch.mul((2. * y - 1.), deltaZ) #if y=0 z - delta_z, if y=1 z + delta_z
			xSwap = vae.decode(zSwap) #decode and classify  #### DETACH? cause update VAE at the bottom!!!!!!!! <<<<<<<<
			predY = classer(xSwap)
			switchedLabels = 1 - y  #classification loss with new labels + PENALTY L2 on the deltaZ
			regTerm = torch.norm(deltaZ, p=opts.p)
			deltaZLoss = bce(predY, switchedLabels).mean() + opts.rho * regTerm
			deltaZLoss.backward()
			optimizer_DZ.step()


			if i%100 == 1:
				print '[%d, %d] rec: %0.5f, kl: %0.5f, class: %0.5f, DZ: %0.5f, |DZ|: %0.5f, time: %0.3f' % \
		 	(e, i, bceLoss.mean().data[0], klLoss.mean().data[0], classLoss.mean().data[0],\
		 		deltaZLoss.mean().data[0], regTerm.mean().data[0], time() - TIME)

		 		vaeLosses['bce'].append(bceLoss.mean().data[0])
				vaeLosses['kl'].append(klLoss.mean().data[0])
				classerLosses['bce'].append(classLoss.mean().data[0])
				deltaZLosses['total'].append(deltaZLoss.mean().data[0])
				deltaZLosses['reg'].append(regTerm.mean().data[0])

		 

		#save params
		print 'saving params...'
		vae.save_params(exDir)
		classer.save_params(exDir)
		torch.save(deltaZ, join(exDir, 'deltaZ'))

		#generate samples after each epoch		
		vae.eval()
		print 'saving a set of samples'
		z = vae.sample_z(opts.batchSize, sig=opts.sig)
		if vae.useCUDA:
			samples = vae.decode(z).cpu()
		else:
			samples = vae.decode(z)

		save_image(samples.data, join(exDir,'epoch'+str(e)+'.png'))

		#check reconstructions after each 10 epochs
		vae.eval()
		classer.eval()

		testIter = iter(trainLoader)
		trainData = testIter.next()
		xTrain, yTrain = prep_data(trainData, classer.useCUDA)

		testIter = iter(testLoader)
		testData = testIter.next()
		xTest, yTest = prep_data(testData, classer.useCUDA)
		
		####### eval VAE #######
		recTest, outMu, outLogVar = vae(xTest) 
		bceLossTest, klLossTest = vae.loss(recTest, xTest, outMu, outLogVar)
		save_image(xTest.data, join(exDir,'input.png'))
		save_image(recTest.data, join(exDir,'output_'+str(e)+'.png'))

		####### eval CLASSER #######
		yPredTrain = classer(xTrain)
		yPredTest = classer(xTest)
		classTrain = score(yPredTrain, yTrain).data[0]
		classTest = score(yPredTest, yTest).data[0]

		vaeLosses['test_bce'].append(bceLossTest.mean().data[0])
		vaeLosses['test_kl'].append(bceLossTest.mean().data[0])
		classerLosses['train_acc'].append(classTrain)
		classerLosses['test_acc'].append(classTest)

		viz_DZ(deltaZ.data[0].cpu().numpy(), exDir)


		if e>0:
			plot_norm_losses(vaeLosses, exDir, e, title='VAE')
			plot_losses(classerLosses, exDir, e, title='CLASSER')
			plot_losses(deltaZLosses, exDir, e, title='DELTAZ')
	

		#Evaluate the results:
		eval_results(vae, classer, deltaZ, testLoader, exDir)

	inception_score(vae, classer, deltaZ, testLoader, exDir)




	

