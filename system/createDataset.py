#from system.utils.data_utils import read_data_new,change_data,read_data,RandomShuffledata
from utils.data_utils import read_data_new,change_data,read_data,RandomShuffledata
import numpy as np
import torch

class DataSet(object):
    def __init__(self, name,num_clients,writedir):
        #数据集名称
        self.dataset=name
        #要分成的客户端个数
        self.num_clients=num_clients
        #数据集文件，“m1","m2",.....
        self.filedir="origin"
        #写入数据地址。保证每个client的数据不小于一个batch_size
        self.writedir=writedir

    def writetxt(self):
        #读取原始数据
        print("Information---------------------reading data-----------------------------------------")
        train_shuffledata = read_data_new(self.dataset, self.num_clients,self.filedir, is_train=True)
        test_shuffledata = read_data_new(self.dataset, self.num_clients,self.filedir, is_train=False)
        #写入数据
        for idx in range(self.num_clients):
            train_data = train_shuffledata[idx]
            print(f"-----------------------write train txt,train_data type is,client {idx}", type(train_data), "length is :", len(train_data))
            writedata = change_data(train_data)
            data_arrays = {}
            for key, value in writedata.items():
                data_arrays[key] = np.array(value)
            trainfile = "/Users/alice/Desktop/FedJSND/dataset/"+self.writedir+"/train/train" + str(idx) + "_.npz"
            with open(trainfile, 'wb') as f:
                np.savez(f, data=data_arrays)
            print("------------------------write test", trainfile, " txt success")

            test_data = test_shuffledata[idx]
            print(f"-------------------------write test txt,test_data type is,client {idx}", type(test_data), "length is :", len(test_data))
            writedata = change_data(test_data)
            data_arrays = {}
            for key, value in writedata.items():
                data_arrays[key] = np.array(value)
            testfile = "/Users/alice/Desktop/FedJSND/dataset/"+self.writedir+"/test/test" + str(idx) + "_.npz"
            with open(testfile, 'wb') as f:
                np.savez(f, data=data_arrays)
            print("------------------------write test", testfile, " txt success")

    def get_data(dataset, num_clients,filedir, is_train):
        '''
        重新分配节点的数据(self.dataset, self.num_clients, is_train=True)
        '''
        alldata = []
        label = []
        for i in range(20):
            # data={'x':numpy.ndarray,,'y':numpy.ndarray}
            # data['x'].shape is(1972, 1, 28, 28),client_data_num=1972
            # data['y'].shape is (1972, )
            data = read_data(dataset, i,filedir, is_train)
            # print("read_data origin data", data['x'].shape,data['y'].shape)
            # 变换成张量形式
            X = torch.Tensor(data['x']).type(torch.float32)
            Y = torch.Tensor(data['y']).type(torch.int64)
            # 整体汇集成一个元组list，特征和标签数据相对应
            data = [(x, y) for x, y in zip(X, Y)]
            labeldata = [y.tolist() for x, y in zip(X, Y)]
            # 所有数据拷贝到alldata
            for k in range(len(data)):
                alldata.append(data[k])
                label.append(labeldata[k])

        # print("alldata is ",len(alldata),"example is ",alldata[0])
        # 将数据随机分割成num_clients个客户端数据
        shuffledata = RandomShuffledata(alldata, num_clients)

        # print("shuffledata",shuffledata)

        return shuffledata

    def RandomShuffledata(alldata, client_nums):
        # print(" ------------------------------------------------------------------------------------------data  RandomShuffledata -----------------------------")

        '''
        #将一个list alldata 分割成client_nums个子list
        '''
        if len(alldata) <= client_nums:
            print("Shuffledata client_nums ERROR", "data length is ", len(alldata), "client num is ", client_nums)
        # Shuffle A randomly
        random.shuffle(alldata)
        each_client_datanums_list = generate_Randomnums(client_nums, len(alldata))

        allindex = list(range(len(alldata)))
        selectedindex = []
        shuffledata = []
        i = 0
        for num in each_client_datanums_list:
            client_data = []
            client_index = select_k_unique_elements(allindex, num, selectedindex)
            i += 1
            for index in client_index:
                client_data.append(alldata[index])
                # 已经选过的index放回数据中，下次则不再选
                selectedindex.append(index)
            # print("client ", i + 1, "data:", num, "alldata is:", len(alldata), len(client_data),type(client_data[0]))
            shuffledata.append(client_data)
        # print("shuffle",shuffledata[0])

        # 判定result数据
        sums = 0
        for clienti in shuffledata:
            sums += len(clienti)
        if sums != len(alldata) or len(shuffledata) != len(each_client_datanums_list):
            if len(shuffledata) == len(each_client_datanums_list) + 1:
                print("ssssss", len(shuffledata), len(each_client_datanums_list))

            else:
                print("Shuffledata  SHUFFLE ERROR: ", "shuffledata length is ", len(shuffledata), "but client_nums is ",
                      len(each_client_datanums_list), client_nums)
        else:
            print("shuffle data ", len(shuffledata), "个 list ,total length is ", len(alldata))
            # print("result[0][0] example is ",result[0][0])

        return shuffledata


data=DataSet("mnist-0.1-npz",1000,"m1000")
data.writetxt()



