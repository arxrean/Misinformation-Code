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
	parser.add_argument('--hid_dim', type=int, default=32)
	parser.add_argument('--lstm_layer', type=int, default=2)
	parser.add_argument('--lr', type=float, default=1e-3)
	opt = parser.parse_args()

	return opt


class MySet(Dataset):
	def __init__(self, opt, df, mode='train'):
		self.opt = opt
		self.df = df
		self.mode = mode
		self.tokenizer = BertTokenizer.from_pretrained(self.opt.bert_name)
		self.dataset = opt.data_path.split('/')[-1].replace('.csv', '')
		self.transform = self.get_img_transform(opt, self.mode)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		text = row['text']
		label = row['reliability']
		encoding = self.tokenizer.encode_plus(
			text,
			add_special_tokens=True,
			max_length=self.opt.max_len,
			return_token_type_ids=False,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_tensors='pt',
		)

		img_feat = self.transform(Image.open(row['imgpath']).convert('RGB'))

		return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten(), img_feat, label

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
		self.args = opt
		emb_dim = opt.hid_dim
		self.bert = BertModel.from_pretrained(self.args.bert_name, return_dict=True)
		self.bert_fc = nn.Linear(768, 32)

		self.hidden_size = opt.hid_dim
		self.vgg19 = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).children())[:-1])

		channel_in = 1
		filter_num = 20
		window_size = [1, 2, 3, 4]
		self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
		self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

		self.image_fc1 = nn.Linear(512,  self.hidden_size)

		self.class_classifier = nn.Sequential()
		self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))

		for p in self.vgg19.parameters():
			p.requires_grad = False

	def init_hidden(self, batch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
				to_var(torch.zeros(1, batch_size, self.lstm_size)))

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
		#x = F.avg_pool1d(x, x.size(2)).squeeze(2)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)

		return x

	def forward(self, text, mask, image):
		### IMAGE #####
		image = self.vgg19(image)
		image = F.leaky_relu(self.image_fc1(image.mean(-1).mean(-1)))
		
		##########CNN##################
		text = self.bert(text, attention_mask=mask).pooler_output
		text = self.bert_fc(text)

		text_image = torch.cat((text, image), 1)

		### Fake or real
		class_output = self.class_classifier(text_image)
		## Domain (which Event )
		# reverse_feature = grad_reverse(text_image)
		# domain_output = self.domain_classifier(reverse_feature)

		return class_output

# python spotfake_baseline.py --epoches 20 --num_workers 4 --batch_size 128  --lr 1e-4 --data_path ./repo/recovery.csv
if __name__ == '__main__':
	opt = get_parser()

	csv = pd.read_csv(opt.data_path)
	csv = csv[csv['type']=='news']
	train_df, val_df = train_test_split(
		csv, test_size=0.5, random_state=opt.seed)

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

	train_set, val_set = MySet(opt, train_df, 'train'), MySet(opt, val_df, 'val')
	trainloader = DataLoader(train_set, batch_size=opt.batch_size,
							 shuffle=True, num_workers=opt.num_workers, drop_last=True)
	valloader = DataLoader(val_set, batch_size=opt.batch_size,
						   shuffle=False, num_workers=opt.num_workers)

	model = MyModel(opt).cuda()

	loss_func = nn.CrossEntropyLoss()

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
			texts, masks, imgs, labels = texts.cuda(), masks.cuda(), imgs.float().cuda(), labels.long().cuda()
			output  = model(texts, masks, imgs)
			loss = loss_func(output, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			pbar.set_postfix({'loss:': '{:.4f}'.format(loss.item())})

	model, pred, gt = model.eval(), [], []
	with torch.no_grad():
		for step, pack in enumerate(valloader):
			texts, masks, imgs, labels = pack
			texts, masks, imgs, labels = texts.cuda(), masks.cuda(), imgs.float().cuda(), labels.long().cuda()
			output = model(texts, masks, imgs)
			pred.append(output.cpu().numpy())
			gt.append(labels.cpu().numpy())

	print(classification_report(np.concatenate(gt), np.argmax(np.concatenate(pred, 0), 1)))
	torch.save(model.state_dict(), './save/spotfake.pth.tar')