import os
import pdb
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import utils


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='tmp')
	parser.add_argument('--seed', type=int, default=42)
	# data
	parser.add_argument('--data_path', type=str, default='./repo/recovery.csv')
	parser.add_argument('--max_len', type=int, default=128)
	parser.add_argument('--img_size', type=int, default=256)
	parser.add_argument('--crop_size', type=int, default=224)
	parser.add_argument('--max_word', type=int, default=5000)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--bert_name', type=str, default='bert-base-cased')
	# model
	parser.add_argument('--embed_model', type=str, default='resnet18')
	parser.add_argument('--epoches', type=int, default=20)
	parser.add_argument('--word_dim', type=int, default=32)
	parser.add_argument('--lstm_layer', type=int, default=2)
	parser.add_argument('--lr', type=float, default=1e-3)
	opt = parser.parse_args()

	return opt


class MySet(Dataset):
	def __init__(self, opt, df, mode='train'):
		self.opt = opt
		self.df = df
		self.mode = mode
		self.dataset = opt.data_path.split('/')[-1].replace('.csv', '')
		self.transform = self.get_img_transform(opt, self.mode)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		text = row['text']
		label = row['reliability']
		text = [self.word2idx.get(x, self.opt.max_word+1) for x in text.split()]
		if len(text) > self.opt.max_len:
			text = text[:self.opt.max_len]
			mask = [1] * self.opt.max_len
		else:
			mask = [1]*len(text) + [0] * (self.opt.max_len-len(text))
			text = text + [0] * (self.opt.max_len-len(text))

		img_feat = self.transform(Image.open(row['imgpath']).convert('RGB'))

		return torch.LongTensor(text), torch.LongTensor(mask), img_feat, label

	def get_img_transform(self, opt, mode):
		normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		if mode == 'train':
			return transforms.Compose([
				transforms.Resize((opt.img_size, opt.img_size)),
				transforms.RandomCrop((opt.crop_size, opt.crop_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		else:
			return transforms.Compose([
				transforms.Resize((opt.crop_size, opt.crop_size)),
				transforms.ToTensor(),
				normalize,
			])


class MyModel(nn.Module):
	def __init__(self, opt):
		super(MyModel, self).__init__()
		self.opt = opt
		self.vgg16 = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).children())[:-1])
		self.emb = nn.Embedding(self.opt.max_word+2, self.opt.word_dim)
		self.lstm_layer = nn.LSTM(self.opt.word_dim, self.opt.word_dim, self.opt.lstm_layer, batch_first=True, bidirectional=True)
		self.text_encode = nn.Sequential(
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim),
			nn.Tanh()
		)
		self.image_encode = nn.Sequential(
			nn.Linear(512, 1024),
			nn.Tanh(),
			nn.Linear(1024, self.opt.word_dim),
			nn.Tanh(),
		)
		self.merge_layer = nn.Sequential(
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim*2),
			nn.Tanh(),
		)
		self.latent_mean = nn.Linear(self.opt.word_dim*2, self.opt.word_dim*2)
		self.latent_var = nn.Linear(self.opt.word_dim*2, self.opt.word_dim*2)

		self.cls = nn.Sequential(
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim*2),
			nn.Tanh(),
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim),
			nn.Tanh(),
			nn.Linear(self.opt.word_dim, 2),
		)

		self.text_decode = nn.Sequential(
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim),
			nn.Tanh(),
		)
		self.lstm_layer_decode = nn.LSTM(self.opt.word_dim, self.opt.word_dim, self.opt.lstm_layer, batch_first=True)
		self.text_decode2 = nn.Linear(self.opt.word_dim, self.opt.max_word+2)

		self.img_decode = nn.Sequential(
			nn.Linear(self.opt.word_dim*2, self.opt.word_dim),
			nn.Tanh(),
			nn.Linear(self.opt.word_dim, 1024),
			nn.Tanh(),
			nn.Linear(1024, 512),
			nn.Sigmoid()
		)

		for p in self.vgg16.parameters():
			p.requires_grad = False

	def forward(self, texts, masks, imgs):
		texts = self.text_encode(self.lstm_layer(self.emb(texts))[0][:, -1])
		img_embeds = self.vgg16(imgs).mean(-1).mean(-1)
		imgs = self.image_encode(img_embeds)
		merge = self.merge_layer(torch.cat([texts, imgs], -1))
		mean = self.latent_mean(merge)
		log_var = self.latent_var(merge)
		sample = self.sampling(mean, log_var)

		output = self.cls(sample)

		text_decode = self.text_decode2(self.lstm_layer_decode(self.text_decode(sample).unsqueeze(1).repeat(1, self.opt.max_len, 1))[0])
		img_decode = self.img_decode(sample)

		return output, text_decode, img_decode, mean, log_var, img_embeds

	def sampling(self, mean, var):
		epsilon = torch.normal(0., 0.01, size=mean.shape).cuda()

		return mean + torch.exp(0.5 * var) * epsilon

# python mvae_baseline.py --epoches 200 --num_workers 4 --batch_size 128  --lr 1e-4 --data_path ./repo/recovery.csv
if __name__ == '__main__':
	opt = get_parser()

	csv = pd.read_csv(opt.data_path)
	train_df, val_df = train_test_split(
		csv, test_size=0.3, random_state=opt.seed)
	word2idx, idx2word = utils.get_word_dict(opt, train_df, 'text')

	# upsample
	majority, minority = train_df[train_df['reliability']
								  == 0], train_df[train_df['reliability'] == 1]
	if len(majority) < len(minority):
		majority, minority = minority, majority
	minority = minority.sample(n=len(majority), replace=True)
	train_df = pd.concat([majority, minority], axis=0).reset_index(drop=True)
	val_df = val_df.reset_index(drop=True)

	# summary
	print('min words: {} max words: {}'.format(csv['text'].str.split(
	).str.len().min(), csv['text'].str.split().str.len().max()))
	print('words distribution: \n{}'.format(
		pd.cut(csv['text'].str.split().str.len(), 5).value_counts()))

	train_set, val_set = MySet(opt, train_df, 'train'), MySet(opt, val_df, 'val')
	train_set.word2idx, train_set.idx2word = word2idx, idx2word
	val_set.word2idx, val_set.idx2word = word2idx, idx2word
	trainloader = DataLoader(train_set, batch_size=opt.batch_size,
							 shuffle=True, num_workers=opt.num_workers, drop_last=True)
	valloader = DataLoader(val_set, batch_size=opt.batch_size,
						   shuffle=False, num_workers=opt.num_workers)

	model = MyModel(opt).cuda()

	loss_func = nn.CrossEntropyLoss()
	loss_func2 = nn.MSELoss()

	optimizer = AdamW(model.parameters(), lr=opt.lr)

	total_training_steps = len(trainloader) * opt.epoches
	warmup_steps = total_training_steps // 5
	scheduler = get_linear_schedule_with_warmup(
	  optimizer,
	  num_warmup_steps=warmup_steps,
	  num_training_steps=total_training_steps
	)

	for epoch in range(opt.epoches):
		model = model.train()
		pbar = tqdm(trainloader, desc='train epoch: {}'.format(epoch))
		for step, pack in enumerate(pbar):
			texts, masks, imgs, labels = pack
			texts, masks, imgs, labels = texts.cuda(), masks.cuda(), imgs.float().cuda(), labels.cuda()
			output, text_decode, img_decode, mean, log_var, img_embeds = model(texts, masks, imgs)
			loss = loss_func(output, labels) + loss_func(text_decode.reshape(-1, opt.max_word+2), texts.detach().reshape(-1)) + \
			loss_func2(img_decode, img_embeds.detach()) - 0.5 * torch.sum(torch.mean(1 + log_var - mean**2 - torch.exp(log_var), axis=-1))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			pbar.set_postfix({'loss:': '{:.4f}'.format(loss.item())})

	model, pred, gt = model.eval(), [], []
	with torch.no_grad():
		for step, pack in enumerate(valloader):
			texts, masks, imgs, labels = pack
			texts, masks, imgs, labels = texts.cuda(), masks.cuda(), imgs.float().cuda(), labels.cuda()
			output, text_decode, img_decode, mean, log_var, img_embeds = model(texts, masks, imgs)
			pred.append(output.cpu().numpy())
			gt.append(labels.cpu().numpy())

	print(classification_report(np.concatenate(gt), np.argmax(np.concatenate(pred, 0), 1)))
	torch.save(model.state_dict(), './save/mvae.pth.tar')