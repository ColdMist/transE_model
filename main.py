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
import time
class TransE(nn.Module):

    # initialize variables defined in the class
    def __init__(self):
        super(TransE,self).__init__()
        self.margin = 1.0
        self.embedding_dim = 50
        self.n_batch = 10000
        self.batch_size = 1
        self.n_epoch = 1000
        self.n_entity = 1
        self.n_relation = 1
        self.l_rate = 0.01
        self.train_triple = 1
        self.test_triple = 1
        self.valid_triple = 1
        self.entity_id = 1
        self.relation_id = 1
        self.n_train = 1
        self.valid = 1
        self.n_test = 1
        self.read_data()
        self.entity_vec = nn.Embedding(self.n_entity, self.embedding_dim)
        #self.entity_vec.requires_grad = True
        self.relation_vec = nn.Embedding(self.n_relation, self.embedding_dim)
        #self.relation_vec.requires_grad = True
        self.init_weights()
        self.device = torch.device('cpu')

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
        train_triple = np.array(train_triple_temp) #Why again creating an array
        self.n_train = int(train_triple[0])
        train_triple = np.array(np.delete(train_triple, 0)) #why deleting the triples

        # Save training triples in vector
        train_triple_final = np.empty([self.n_train, 3], dtype=int)
        for i in range(0, self.n_train):
            train_triple_final[i] = train_triple[i]

        self.train_triple = train_triple_final

        # Save validation triples in vector
        valid_triple_file = valid_file.read()
        valid_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in valid_triple_file.splitlines()]
        valid_triple = np.array(valid_triple_temp)
        self.n_valid = int(valid_triple[0])
        valid_triple = np.delete(valid_triple, 0)

        valid_triple_final = np.empty([self.n_valid, 3], dtype=int)
        for i in range(0, self.n_valid):
            valid_triple_final[i] = valid_triple[i]

        self.valid_triple = valid_triple_final

        # Save testing triples in vector
        test_triple_file = test_file.read()
        test_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in test_triple_file.splitlines()]
        test_triple = np.array(test_triple_temp)
        self.n_test = int(test_triple[0])
        test_triple = np.delete(test_triple, 0)

        test_triple_final = np.empty([self.n_test, 3], dtype=int)
        for i in range(0, self.n_test):
            test_triple_final[i] = test_triple[i]

        self.test_triple = test_triple_final

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

    def plot_results(self, x, y):
        fig = plt.figure()
        plt.plot(x, y, 'ro')
        plt.axis([0, np.amax(x), 0, np.amax(y)])
        plt.show()
        fig.savefig('Convergence.png')

    def save_vec(self):

        fe = open("entity2vecFinal", "w")
        fr = open("relation2vecFinal", "w")

        for i in range(0, self.relation_vec.num_embeddings):
            for ii in range(0, self.embedding_dim):
                fr.write(str(self.relation_vec.weight.data[i, ii].numpy()) + '\t')
            fr.write("\n")

        for i in range(0, self.entity_vec.num_embeddings):
            for ii in range(0, self.embedding_dim):
                fe.write(str(self.entity_vec.weight.data[i, ii].numpy()) + '\t')
            fe.write("\n")

        fe.close()
        fr.close()

    def init_weights(self):
        #nn.init.xavier_uniform(self.entity_vec.weight.data)
        #nn.init.xavier_uniform(self.relation_vec.weight.data)
        self.entity_vec.weight.data = torch.tensor(np.random.uniform(-1 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim), [self.n_entity, self.embedding_dim]))
        #self.entity_vec.weight.to(torch.device('cuda'))
        self.relation_vec.weight.data = torch.tensor(np.random.uniform(-1 / np.sqrt(self.embedding_dim), 1 / np.sqrt(self.embedding_dim),
                                              [self.n_relation, self.embedding_dim]))
        #self.relation_vec.weight.to(torch.device('cuda'))

    def normali(self, p_idx, n_idx):
        h = p_idx[:, 0]
        t = p_idx[:, 1]
        r = p_idx[:, 2]

        hn = n_idx[:, 0]
        tn = n_idx[:, 1]
        rn = n_idx[:, 2]

        self.entity_vec.weight.data[h] = F.normalize(self.entity_vec.weight.data[h],p=1,dim=1)
        self.entity_vec.weight.data[t] = F.normalize(self.entity_vec.weight.data[t], p=1, dim=1)
        #self.relation_vec.weight.data[r] = F.normalize(self.relation_vec.weight.data[r], p=1, dim=1)

        self.entity_vec.weight.data[hn] = F.normalize(self.entity_vec.weight.data[hn], p=1, dim=1)
        self.entity_vec.weight.data[tn] = F.normalize(self.entity_vec.weight.data[tn], p=1, dim=1)
        #self.relation_vec.weight.data[rn] = F.normalize(self.relation_vec.weight.data[rn], p=1, dim=1)

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

            selected_head_vec = self.entity_vec(torch.tensor(np.array(selected_head)))
            selected_tail_vec = self.entity_vec(torch.tensor(np.array(selected_tail)))
            selected_relation_vec = self.relation_vec(torch.tensor(np.array(selected_relation)))

            ht = self.entity_vec(torch.tensor(h))
            tt = self.entity_vec(torch.tensor(t))
            rt = self.relation_vec(torch.tensor(r))

            ht_tail = self.entity_vec(torch.tensor(th))
            tt_tail = self.entity_vec(torch.tensor(tt1))
            rt_tail = self.relation_vec(torch.tensor(tr))

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

    def loss_func(self,pos_score,neg_score):
        yy = Variable(torch.tensor([-1.0], dtype = torch.float64))
        #yy=yy.cuda()
        #print(type(yy))
        #exit()
        criterion = nn.MarginRankingLoss(self.margin)
        p = torch.tensor([100.5, 10], dtype = torch.float64)
        #p = p.type(torch.cuda.FloatTensor)
        n = torch.tensor([1, 1], dtype = torch.float64)
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

    def forward(self, p_idx, n_idx):
        h = p_idx[:, 0]
        t = p_idx[:, 1]
        r = p_idx[:, 2]

        hn = n_idx[:, 0]
        tn = n_idx[:, 1]
        rn = n_idx[:, 2]

        ht = self.entity_vec(torch.tensor(h))
        #ht.to(torch.device('cuda'))
        tt = self.entity_vec(torch.tensor(t))
        #tt.to(torch.device('cuda'))
        rt = self.relation_vec(torch.tensor(r))
        #rt.to(torch.device('cuda'))

        htn = self.entity_vec(torch.tensor(hn))
        #htn.to(torch.device('cuda'))
        ttn = self.entity_vec(torch.tensor(tn))
        #ttn.to(torch.device('cuda'))
        rtn = self.relation_vec(torch.tensor(rn))
        #rtn.to(torch.device('cuda'))

        negative_score = model.score_func(htn, rtn, ttn)
        positive_score = model.score_func(ht, rt, tt)
        loss = self.loss_func(positive_score, negative_score)
        return loss



def run(model):
    #model.to(torch.device("cuda:0"))
    train_triples = model.train_triple
    n_train = np.shape(train_triples)[0]
    n_batch = model.n_batch
    n_epoch = model.n_epoch

    l_rate = model.l_rate
    optimiser = optim.SGD(model.parameters(), lr=l_rate)
    gg = model.parameters()
    Total_loss = np.zeros([n_epoch])
    idx = np.arange(0, model.n_entity)
    inp = np.arange(0, n_epoch)
    for epoch in range(0, n_epoch):
        t1 = time.time()
        batch_loss = Variable(torch.tensor([0.0], dtype = torch.float64))
        np.random.shuffle(train_triples)
        batch_triples = np.copy(np.array_split(train_triples, n_batch))
        #batch_triples[0] = torch.from_numpy(batch_triples[0])
        #batch_triples[1] = torch.from_numpy(batch_triples[1])
        #batch_triples[2] = torch.from_numpy(batch_triples[2])
        #batch_triples[0].to(torch.device('cuda'))
        #batch_triples[1].to(torch.device('cuda'))
        #batch_triples[2].to(torch.device('cuda'))
        #print(type(batch_triples[0]))
        #batch_triples = torch.from_numpy(batch_triples)
        for batch_n in  range(0,n_batch):
            np.random.shuffle(idx)
            temp = np.shape(batch_triples[batch_n])[0]
            batch_index = np.copy(idx[0:temp])
            selected_batch = np.copy(batch_triples[batch_n])
            selected_batch = torch.from_numpy(selected_batch)
            selected_batch.to(model.device)
            p_idx = np.copy(selected_batch)
            #print(type(selected_batch))
            if batch_n % 2 == 0:
                h_corupted = np.copy(batch_index)
                n_idx = np.copy(p_idx)
                n_idx[:, 0] = np.copy(h_corupted)
            else:
                t_corupted = np.copy(batch_index)
                n_idx = np.copy(p_idx)
                n_idx[:, 1] = np.copy(t_corupted)
            model.normali(p_idx, n_idx)
            optimiser.zero_grad()
            loss = model(p_idx,n_idx)
            batch_loss = batch_loss + loss * torch.tensor(np.shape(batch_triples[batch_n])[0], dtype = torch.float64)
            loss.backward()
            optimiser.step()
            #for i in model.parameters():
            #    if i.requires_grad:
            #        print(i.grad)
            #model.normali(p_idx, n_idx)
            t2 = time.time()
        print('time taken per epoch: ', t2-t1)
        Total_loss[epoch] = batch_loss.data.numpy() / n_train
        print('epoch {}, loss {}'.format(epoch, batch_loss/n_train))
    model.save_vec()
    model.plot_results(inp, Total_loss)
    model.testing()


#config = MyConf(1,100,100,1,30,1,1,1,1,1,1,1,1)
model = TransE()
#model.double()
#use_gpu = torch.cuda.is_available()
#if use_gpu:
#    model = model.cuda()
t0 = time.time()
#model.to("cuda")
#model.to("cuda:0")
#cudnn.benchmark = True
run(model)
t1 = time.time()
print(t1-t0)

