import copy
import numpy as np
import torch
import pandas as pd
import time
from flcore.clients.clientALA import *
from utils.data_utils import read_client_data
from methods.distance import jensen_shannon_distance
from threading import Thread


class FedALA(object):
    def __init__(self, args, times,filedir):
        #定义method
        self.method=None
        #m1,m2,None
        self.filedir=filedir












        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_clients(args, clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        colum_value = []
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #self.evaluate()
                res = self.evaluate(i)
                colum_value.append(res)

            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # --------------acc
        colum_name = ["method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            redf.loc[len(redf) + 1] = colum_value[i]
        path = "/Users/alice/Desktop/FedJSND/res/ala.csv"
        redf.to_csv(path, mode='a', header=False)


    def set_clients(self, args, clientObj):
        #整个数据集的标签向量
        alllabelvectors = []
        #用来存放id:labellist
        labeldict={}
        for i in range(self.num_clients):
            #---------
            train_data,train_label = read_client_data(self.dataset, i,self.filedir, is_train=True)
            test_data,test_label = read_client_data(self.dataset, i,self.filedir, is_train=False)
            #单个client的标签向量
            client_label=train_label
            for j in range(len(test_label)):
                client_label[j]=test_label[j]+train_label[j]

            # labeldict[i]=client_label
            #将所有的client 的label汇总------------------
            if i==0:
                alllabelvectors=client_label
            else:
                for j in range(len(client_label)):
                    alllabelvectors[j] += client_label[j]
            #print("INFormation---------set client ,num_client,client_label", self.num_clients, client_label, alllabelvectors)

            client = clientObj(args,
                            id=i,
                            filedir=self.filedir,
                            client_label=client_label,
                            train_samples=len(train_data),
                            test_samples=len(test_data))
            self.clients.append(client)
        #---------
        for c in self.clients:
            # ---------
            c.distance=jensen_shannon_distance(c.client_label, alllabelvectors)
            print(f"INFormation-----------------------------------init client {c.id} JS distance is {c.distance}-------------------------------")




    # def set_clients(self, args, clientObj):
    #
    #     for i in range(self.num_clients):
    #         # ---------
    #         train_data, train_label = read_client_data(self.dataset, i, is_train=True)
    #         test_data, test_label = read_client_data(self.dataset, i, is_train=False)
    #         print(f"Information--------------------client {i} data length is :",len(train_data),len(test_data))
    #         client = clientObj(args,
    #                            id=i,
    #                            client_label=None,
    #                            train_samples=len(train_data),
    #                            test_samples=len(test_data))
    #         self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            print(f"Information--------------------client {client.id} data length is :{client.train_samples}")
            client.local_initialization(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self,group, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method,group,train_loss,test_acc,test_auc,np.std(accs),np.std(aucs)]
