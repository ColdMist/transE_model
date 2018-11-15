## instead

""""
WN18
mean rank is:
2741.9396
mean hit at 1 is:
0.0262
mean hit at 3 is:
0.0759
mean hit at 10 is:
0.1449
Model parameters:
Embedding dim
20
Lr
0.01
"""

"""
mean rank is:
2920.6347
mean hit at 1 is:
0.0317
mean hit at 3 is:
0.0778
mean hit at 10 is:
0.1498
Model parameters:
Embedding dim
20
Lr
0.01
reg: 1
"""

"""
mean rank is:
2747.831
mean hit at 1 is:
0.0323
mean hit at 3 is:
0.0801
mean hit at 10 is:
0.1521
Model parameters:
Embedding dim
20
Lr
0.01
reg: 0.1
"""

"""
WN or FB???
mean rank is:
420.2804
mean hit at 1 is:
0.1212
mean hit at 3 is:
0.4028
mean hit at 10 is:
0.605
Model parameters:
Embedding dim
20
Lr
0.1
reg: 0.1
Margin 2
2000
"""


import numpy as np
import os
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import statistics as st
import torch.nn.functional as F
import numpy.matlib as mlib
import matplotlib.pyplot as plt

class TransE(nn.Module):

    # initialize variables defined in the class
    def __init__(self):
        super(TransE,self).__init__()
        self.margin = 2.0
        self.embedding_dim = 20
        self.n_batch = 100
        self.batch_size = 1
        self.n_epoch = 5
        self.n_entity = 1
        self.n_relation = 1
        self.l_rate = 0.1
        self.train_triple = 1
        self.test_triple = 1
        self.valid_triple = 1
        self.entity_id = 1
        self.relation_id = 1
        self.n_train = 1
        self.valid = 1
        self.n_test = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.read_data()
        self.entity_vec = nn.Embedding(self.n_entity, self.embedding_dim)
        #self.entity_vec.requires_grad = True
        self.relation_vec = nn.Embedding(self.n_relation, self.embedding_dim)
        self.error = nn.Embedding(self.n_train, 1) #Variable(torch.tensor(np.random.uniform(-1 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim),
                              #[self.n_train, 1])), requires_grad=True)
        self.init_weights()

        # Read training, validation and testing data (triples)

    def read_data(self):
        # Get Path of Data
        file = "/FB15K/train2id.txt"
        path_train = os.getcwd() + file
        file = "/FB15K/valid2id.txt"
        path_test = os.getcwd() + file
        file = "/FB15K/test2id.txt"
        path_valid = os.getcwd() + file
        file = "/FB15K/entity2id.txt"
        path_entity = os.getcwd() + file
        file = "/FB15K/relation2id.txt"
        path_relation = os.getcwd() + file

        # Open Data Files
        train_file = open(path_train, "r")
        valid_file = open(path_valid, "r")
        test_file = open(path_test, "r")
        entity_file = open(path_entity, "r")
        relation_file = open(path_relation, "r")

        # Read Files of Data
        train_triple_file = train_file.read()
        train_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in train_triple_file.splitlines()]
        train_triple1 = np.array(train_triple_temp)
        self.n_train = int(train_triple1[0])
        train_triple1 = np.array(np.delete(train_triple1, 0))
        o = np.concatenate(train_triple1, axis=0)
        oo = np.reshape(o, (self.n_train, 3))
        train_triple = torch.from_numpy(oo).float().to(self.device)

        self.train_triple = train_triple

        # Save validation triples in vector
        valid_triple_file = valid_file.read()
        valid_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in valid_triple_file.splitlines()]
        valid_triple1 = np.array(valid_triple_temp)
        self.n_valid = int(valid_triple1[0])
        valid_triple1 = np.delete(valid_triple1, 0)

        ov = np.concatenate(valid_triple1, axis=0)
        oov = np.reshape(ov, (self.n_valid, 3))
        valid_triple = torch.from_numpy(oov).float().to(self.device)

        self.valid_triple = valid_triple

        # Save testing triples in vector
        test_triple_file = test_file.read()
        test_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in test_triple_file.splitlines()]
        test_triple1 = np.array(test_triple_temp)
        self.n_test = int(test_triple1[0])
        test_triple1 = np.delete(test_triple1, 0)

        ot = np.concatenate(test_triple1, axis=0)
        oot = np.reshape(ot, (self.n_test, 3))
        test_triple = torch.from_numpy(oot).float().to(self.device)

        self.test_triple = test_triple

        # Save entity-index vector
        entity_tuple_file = entity_file.read()
        entity_tuple_temp = [np.array(s_inner.split('\t')) for s_inner in entity_tuple_file.splitlines()]
        entity_tuple = np.array(entity_tuple_temp)
        self.n_entity = int(entity_tuple[0])
        entity_tuple = np.array(np.delete(entity_tuple, 0))

        entity_tuple_final = np.empty([self.n_entity, 2], dtype='U20')
        for i in range(0, self.n_entity):
            entity_tuple_final[i] = entity_tuple[i]

        self.entity_id = entity_tuple_final

        # Save relation-index in vector
        relation_tuple_file = relation_file.read()
        relation_tuple_temp = [np.array(s_inner.split('\t')) for s_inner in relation_tuple_file.splitlines()]
        relation_tuple = np.array(relation_tuple_temp)
        self.n_relation = int(relation_tuple[0])
        relation_tuple = np.array(np.delete(relation_tuple, 0))

        relation_tuple_final = np.empty([self.n_relation, 2], dtype='U1000')
        for i in range(0, self.n_relation):
            relation_tuple_final[i] = relation_tuple[i]

        self.relation_id = relation_tuple_final
        print("goodbye")

    def plot_results(self, x, y):
        fig = plt.figure()
        plt.plot(x, y, 'ro')
        plt.axis([0, np.amax(x), 0, np.amax(y)])
        #plt.show()
        fig.savefig('Convergence_Loss.png')

    def save_vec(self):

        fe = open("entity2vecFinal", "w")
        fr = open("relation2vecFinal", "w")

        for i in range(0, self.relation_vec.num_embeddings):
            for ii in range(0, self.embedding_dim):
                fr.write(str(self.relation_vec.weight.data[i, ii].to('cpu').numpy()) + '\t')
            fr.write("\n")

        for i in range(0, self.entity_vec.num_embeddings):
            for ii in range(0, self.embedding_dim):
                fe.write(str(self.entity_vec.weight.data[i, ii].to('cpu').numpy()) + '\t')
            fe.write("\n")

        fe.close()
        fr.close()

    def init_weights(self):
        self.entity_vec.weight.data = torch.tensor(np.random.uniform(-1 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim), [self.n_entity, self.embedding_dim]))
        self.relation_vec.weight.data = torch.tensor(np.random.uniform(-1 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim),
                                              [self.n_relation, self.embedding_dim]))
        self.error.weight.data = torch.tensor(np.random.uniform(0 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim),
                              [self.n_train, 1]))

    def normali(self, p_idx, n_idx):
        h = p_idx[:, 0].type(torch.LongTensor)
        t = p_idx[:, 1].type(torch.LongTensor)
        r = p_idx[:, 2].type(torch.LongTensor)

        hn = n_idx[:, 0].type(torch.LongTensor)
        tn = n_idx[:, 1].type(torch.LongTensor)
        rn = n_idx[:, 2].type(torch.LongTensor)

        self.entity_vec.weight.data[h] = F.normalize(self.entity_vec.weight.data[h],p=1,dim=1)
        self.entity_vec.weight.data[t] = F.normalize(self.entity_vec.weight.data[t], p=1, dim=1)

        self.entity_vec.weight.data[hn] = F.normalize(self.entity_vec.weight.data[hn], p=1, dim=1)
        self.entity_vec.weight.data[tn] = F.normalize(self.entity_vec.weight.data[tn], p=1, dim=1)
    def testing(self):
        test_triples = self.test_triple
        n_test = np.shape(test_triples)[0]
        entity = np.array(self.entity_id[:,1], dtype=int)
        relation = np.array(self.relation_id[:,1], dtype=int)
        n_entities = np.shape(entity)[0]

        mean_rank_head = 0
        mean_rank_tail = 0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_10 = 0

        hit_at_1_tail = 0
        hit_at_3_tail = 0
        hit_at_10_tail = 0

        for i in range(0,n_test):
            selected_triple = test_triples[i, :]

            selected_head = selected_triple[0]
            selected_tail = selected_triple[1]
            selected_relation = selected_triple[2]

            head_corrupt_i = mlib.repmat(selected_triple, n_entities, 1)
            tail_corrupt_i = mlib.repmat(selected_triple, n_entities, 1)
            head_corrupt_i[:, 0] = entity
            tail_corrupt_i[:, 1] = entity

            h = head_corrupt_i[:, 0]
            t = head_corrupt_i[:, 1]
            r = head_corrupt_i[:, 2]

            th = tail_corrupt_i[:, 0]
            tt1 = tail_corrupt_i[:, 1]
            tr = tail_corrupt_i[:, 2]

            #selected_head_vec = self.entity_vec(torch.tensor(np.array(selected_head))).to(self.device)
            #selected_tail_vec = self.entity_vec(torch.tensor(np.array(selected_tail))).to(self.device)
            #selected_relation_vec = self.relation_vec(torch.tensor(np.array(selected_relation))).to(self.device)
            #selected_head_vec = self.entity_vec(selected_head.type(torch.cuda.LongTensor))

            #ht = self.entity_vec(torch.cuda.tensor(h))
            #tt = self.entity_vec(torch.cuda.tensor(t))
            #rt = self.relation_vec(torch.cuda.tensor(r))

            #ht_tail = self.entity_vec(torch.cuda.tensor(th))
            #tt_tail = self.entity_vec(torch.cuda.tensor(tt1))
            #rt_tail = self.relation_vec(torch.cuda.tensor(tr))
            selected_head_vec = self.entity_vec(selected_head.type(torch.cuda.LongTensor))
            selected_tail_vec = self.entity_vec(selected_tail.type(torch.cuda.LongTensor))
            selected_relation_vec = self.relation_vec(selected_relation.type(torch.cuda.LongTensor))

            ht = self.entity_vec(torch.tensor(h).type(torch.cuda.LongTensor))
            tt = self.entity_vec(torch.tensor(t).type(torch.cuda.LongTensor))
            rt = self.relation_vec(torch.tensor(r).type(torch.cuda.LongTensor))

            ht_tail = self.entity_vec(torch.tensor(th).type(torch.cuda.LongTensor))
            tt_tail = self.entity_vec(torch.tensor(tt1).type(torch.cuda.LongTensor))
            rt_tail = self.relation_vec(torch.tensor(tr).type(torch.cuda.LongTensor))

            selected_score = self.score_func1(selected_head_vec, selected_relation_vec, selected_tail_vec).data.to('cpu').numpy()
            predicted_score_head = self.score_func(ht, rt, tt).data.to('cpu').numpy()
            predicted_score_tail = self.score_func(ht_tail, rt_tail, tt_tail).data.to('cpu').numpy()

            sorted_prediction_head = np.sort(predicted_score_head, axis= None)
            sorted_prediction_tail = np.sort(predicted_score_tail, axis= None)

            rank_h = np.amin(np.where(sorted_prediction_head == selected_score)[0])
            rank_t = np.amin(np.where(sorted_prediction_tail == selected_score)[0])

            mean_rank_head = mean_rank_head + rank_h
            mean_rank_tail = mean_rank_tail + rank_t

            if rank_h == 1:
                hit_at_1 = hit_at_1 + 1

            if rank_h <= 3:
                hit_at_3 = hit_at_3 + 1

            if rank_h <= 10:
                hit_at_10 = hit_at_10 + 1

            if rank_t == 1:
                hit_at_1_tail = hit_at_1_tail + 1

            if rank_t <= 3:
                hit_at_3_tail = hit_at_3_tail + 1

            if rank_t <= 10:
                hit_at_10_tail = hit_at_10_tail + 1

            print(i)

        mean_rank = (mean_rank_head + mean_rank_tail)/(2*n_test)
        mean_hit_at_1 = (hit_at_1 + hit_at_1_tail) / (2*n_test)
        mean_hit_at_3 = (hit_at_3 + hit_at_3_tail) / (2*n_test)
        mean_hit_at_10 = (hit_at_10 + hit_at_10_tail) / (2*n_test)

        print("mean rank is:")
        print(mean_rank)


        print("mean hit at 1 is:")
        print(mean_hit_at_1)

        print("mean hit at 3 is:")
        print(mean_hit_at_3)

        print("mean hit at 10 is:")
        print(mean_hit_at_10)
    '''
    def testing(self):
        test_triples = self.test_triple
        n_test = test_triples.shape[0]
        entity = np.array(self.entity_id[:,1], dtype=int)
        relation = np.array(self.relation_id[:,1], dtype=int)
        n_entities = np.shape(entity)[0]

        mean_rank_head = 0
        mean_rank_tail = 0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_10 = 0

        hit_at_1_tail = 0
        hit_at_3_tail = 0
        hit_at_10_tail = 0

        for i in range(0,n_test):
            selected_triple = test_triples[i, :]

            selected_head = selected_triple[0]
            selected_tail = selected_triple[1]
            selected_relation = selected_triple[2]

            head_corrupt_i = mlib.repmat(selected_triple, n_entities, 1)
            tail_corrupt_i = mlib.repmat(selected_triple, n_entities, 1)
            head_corrupt_i[:, 0] = entity
            tail_corrupt_i[:, 1] = entity

            h = head_corrupt_i[:, 0]
            t = head_corrupt_i[:, 1]
            r = head_corrupt_i[:, 2]

            th = tail_corrupt_i[:, 0]
            tt1 = tail_corrupt_i[:, 1]
            tr = tail_corrupt_i[:, 2]

            selected_head_vec = self.entity_vec(selected_head.type(torch.cuda.LongTensor))
            selected_tail_vec = self.entity_vec(selected_tail.type(torch.cuda.LongTensor))
            selected_relation_vec = self.relation_vec(selected_relation.type(torch.cuda.LongTensor))

            ht = self.entity_vec(torch.tensor(h).type(torch.cuda.LongTensor))
            tt = self.entity_vec(torch.tensor(t).type(torch.cuda.LongTensor))
            rt = self.relation_vec(torch.tensor(r).type(torch.cuda.LongTensor))

            ht_tail = self.entity_vec(torch.tensor(th).type(torch.cuda.LongTensor))
            tt_tail = self.entity_vec(torch.tensor(tt1).type(torch.cuda.LongTensor))
            rt_tail = self.relation_vec(torch.tensor(tr).type(torch.cuda.LongTensor))

            selected_score = self.score_func1(selected_head_vec, selected_relation_vec, selected_tail_vec).data.numpy()
            predicted_score_head = self.score_func(ht, rt, tt).data.numpy()
            predicted_score_tail = self.score_func(ht_tail, rt_tail, tt_tail).data.numpy()

            sorted_prediction_head = np.sort(predicted_score_head, axis= None)
            sorted_prediction_tail = np.sort(predicted_score_tail, axis= None)

            rank_h = np.amin(np.where(sorted_prediction_head == selected_score)[0])
            rank_t = np.amin(np.where(sorted_prediction_tail == selected_score)[0])

            mean_rank_head = mean_rank_head + rank_h
            mean_rank_tail = mean_rank_tail + rank_t

            if rank_h == 1:
                hit_at_1 = hit_at_1 + 1

            if rank_h <= 3:
                hit_at_3 = hit_at_3 + 1

            if rank_h <= 10:
                hit_at_10 = hit_at_10 + 1

            if rank_t == 1:
                hit_at_1_tail = hit_at_1_tail + 1

            if rank_t <= 3:
                hit_at_3_tail = hit_at_3_tail + 1

            if rank_t <= 10:
                hit_at_10_tail = hit_at_10_tail + 1

            if i % 5000 == 0:
                print(i)

        mean_rank = (mean_rank_head + mean_rank_tail)/(2*n_test)
        mean_hit_at_1 = (hit_at_1 + hit_at_1_tail) / (2*n_test)
        mean_hit_at_3 = (hit_at_3 + hit_at_3_tail) / (2*n_test)
        mean_hit_at_10 = (hit_at_10 + hit_at_10_tail) / (2*n_test)

        print("mean rank is:")
        print(mean_rank)


        print("mean hit at 1 is:")
        print(mean_hit_at_1)

        print("mean hit at 3 is:")
        print(mean_hit_at_3)

        print("mean hit at 10 is:")
        print(mean_hit_at_10)

        print("Model parameters:")

        print("Embedding dim")
        print(self.embedding_dim)

        print("Lr")
        print(self.l_rate)

        print("reg: 0.1")

    def loss_func(self,pos_score,neg_score,error,sel_error):
        err_size = error.shape[0]
        pos_score = pos_score.double().to(self.device)
        neg_score = neg_score.double().to(self.device)
        #error = error.float().to(self.device)

        zero = torch.zeros(err_size,1).float().to(self.device)
        loss_Error = 0.1 * torch.mean(sel_error**2)
        loss_score_positive = torch.mean(pos_score**2)
        right_neg = self.margin * torch.ones(err_size,1).double().to(self.device) - neg_score.double().to(self.device) - error.double().to(self.device)
        loss_score_negative = torch.mean(right_neg ** 2).double().to(self.device)
        loss = loss_Error + loss_score_positive + loss_score_negative
        #print("lossin")
        #print(loss_Error)
        return loss
    '''
    def loss_func(self,pos_score,neg_score):
        yy = Variable(torch.tensor([-1.0], dtype = torch.float64)).to(self.device)
        #yy=yy.cuda()
        #print(type(yy))
        #exit()
        criterion = nn.MarginRankingLoss(self.margin)
        #p = torch.tensor([100.5, 10], dtype = torch.float64).to(self.device)
        #p = p.type(torch.cuda.FloatTensor)
        #n = torch.tensor([1, 1], dtype = torch.float64).to(self.device)
        pos_score = pos_score.double().to(self.device)
        neg_score = neg_score.double().to(self.device)
        #n = n.type(torch.cuda.FloatTensor)
        #pos_score = pos_score.unsqueeze(0)
        #neg_score = neg_score.unsqueeze(0)
        loss = criterion(pos_score, neg_score, yy)
        return loss

    def score_func1(self, h, l, t):
        sb = h + l - t
        asb = torch.abs(sb)
        score1 = torch.sum(asb, 0)
        return score1

    def score_func(self, h, l, t):
        sb = h + l - t
        asb = torch.abs(sb)
        score1 = torch.sum(asb, 1)
        score = torch.norm((h + l - t), 2,1)
        return score1

    def forward(self, p_idx, n_idx,selected_error_idx, selected_error):
        h = p_idx[:, 0].type(torch.cuda.LongTensor)
        t = p_idx[:, 1].type(torch.cuda.LongTensor)
        r = p_idx[:, 2].type(torch.cuda.LongTensor)

        hn = n_idx[:, 0].type(torch.cuda.LongTensor)
        tn = n_idx[:, 1].type(torch.cuda.LongTensor)
        rn = n_idx[:, 2].type(torch.cuda.LongTensor)

        ht = self.entity_vec(h).double().to(self.device)
        tt = self.entity_vec(t).double().to(self.device)
        rt = self.relation_vec(r).double().to(self.device)

        htn = self.entity_vec(hn).double().to(self.device)
        ttn = self.entity_vec(tn).double().to(self.device)
        rtn = self.relation_vec(rn).double().to(self.device)

        negative_score = model.score_func(htn, rtn, ttn)
        positive_score = model.score_func(ht, rt, tt)
        error = self.error(selected_error_idx.type(torch.cuda.LongTensor)).double().to(self.device)
        sel_error = self.error(selected_error.type(torch.cuda.LongTensor)).double().to(self.device)
        loss = self.loss_func(positive_score, negative_score)
        return loss


def run(model):
    train_triples = model.train_triple
    n_train = np.shape(train_triples)[0]
    n_batch = model.n_batch
    n_epoch = model.n_epoch
    perm = torch.randperm(n_train)
    l_rate = model.l_rate
    optimiser = optim.SGD(model.parameters(), lr=l_rate)
    gg = model.parameters()
    #Total_loss = np.zeros([n_epoch])
    Total_loss = torch.zeros([n_epoch], dtype=torch.float64)
    #idx = torch.from_numpy(np.arange(0, model.n_entity)).float().to(model.device)
    idx = torch.from_numpy(np.arange(0, model.n_entity)).float().to(model.device)
    inp = np.arange(0, n_epoch)
    #perm = torch.randperm(n_train)
    #error = model.error(perm.type(torch.LongTensor))
    train_triples = train_triples[perm]

    for epoch in range(0, n_epoch):
        batch_loss = Variable(torch.tensor([0.0], dtype=torch.float64)).to(model.device)

        batch_triples = torch.split(train_triples,int(n_train/n_batch), dim=0)
        error_idx = torch.split(perm, int(n_train / n_batch), dim=0)

        for batch_n in  range(0,n_batch):
            idx = idx[torch.randperm(model.n_entity)]
            temp = batch_triples[batch_n].shape[0]
            batch_index = idx[0:temp]

            selected_batch = batch_triples[batch_n]
            selected_error = perm
            selected_error_idx = error_idx[batch_n]

            p_idx = selected_batch.clone()
            n_idx = selected_batch.clone()

            bin = torch.randint(0, 1, (1,))

            if batch_n % 2 == 0:
                h_corupted = batch_index
                n_idx[:, 0] = h_corupted
            else:
                t_corupted = batch_index
                n_idx[:, 1] = t_corupted

            optimiser.zero_grad()
            loss = model(p_idx,n_idx,selected_error_idx, selected_error)
            batch_loss = batch_loss + loss * torch.tensor(np.shape(batch_triples[batch_n])[0], dtype = torch.float64).to(model.device)
            #batch_loss = Variable(torch.tensor([0.0], dtype=torch.float64)).to(model.device)
            loss.backward()
            #for i in model.parameters():
                #if i.requires_grad:
                    #print(i.grad)
            #print(torch.mean(error**2))
            optimiser.step()
            #print(loss)
            model.normali(p_idx, n_idx)


        Total_loss[epoch] = batch_loss.to(model.device) / n_train
        print('epoch {}, loss {}'.format(epoch, batch_loss/n_train))
    model.save_vec()
    #model.plot_results(inp, Total_loss)
    model.testing()


#config = MyConf(1,100,100,1,30,1,1,1,1,1,1,1,1)
model = TransE()
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()

print("hhh")
run(model)
