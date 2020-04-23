import gc
#from MET4FOFDataReceiver import DataBuffer,DR
import time
#import psutil

import matplotlib.pyplot as plt
from datetime import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold,GridSearchCV

import collections
from numpy.random import seed
from scipy import stats
import timeit

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent, DataStreamAgent
from agentMET4FOF.streams import SineGenerator
from pathlib import Path

import torch
import torchvision
import numpy as np
import pandas as pd
import argparse
import torch.utils.data as data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import random
import copy
import seaborn  as sns
#p=Path(__file__).parent.parent.parent
########################################################################################################################
random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features
########################################################################################################################
class SineGeneratorAgent_Test1(AgentMET4FOF):
        def init_parameters(self, sensor_buffer_size):
            self.stream = SineGenerator()
            self.buffer_size = sensor_buffer_size

        def agent_loop(self):
            if self.current_state == "Running":
                sine_data = self.stream.next_sample()  # dictionary
                now = datetime.now()
                sine_data = {'time': now.second, 'y': sine_data['x']}

                # save data into memory
                self.name="SineGeneratorAgent_Test1"
                self.update_data_memory({'from': self.name, 'data': sine_data})
                # send out buffered data if the stored data has exceeded the buffer size
                if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                    self.send_output(self.memory[self.name])
                    self.memory = {}


class SineGeneratorAgent_Test2(AgentMET4FOF):
    def init_parameters(self, sensor_buffer_size):
        self.stream = SineGenerator()
        self.buffer_size = sensor_buffer_size

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample()  # dictionary
            now = datetime.now()
            sine_data = {'time': now.second, 'y': sine_data['x']}

            # save data into memory
            self.name = "SineGeneratorAgent_Test2"
            self.update_data_memory({'from': self.name, 'data': sine_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

class SineGeneratorAgent_Test3(AgentMET4FOF):
    def init_parameters(self, sensor_buffer_size):
            self.stream = SineGenerator()
            self.buffer_size = sensor_buffer_size

    def agent_loop(self):
            if self.current_state == "Running":
                sine_data = self.stream.next_sample()  # dictionary
                now = datetime.now()
                sine_data = {'time': now.second, 'y': sine_data['x']}

                # save data into memory
                self.name = "SineGeneratorAgent_Test3"
                self.update_data_memory({'from': self.name, 'data': sine_data})
                # send out buffered data if the stored data has exceeded the buffer size
                if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                    self.send_output(self.memory[self.name])
                    self.memory = {}
########################################################################################################################


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))
#########################################################################
class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#########################################################################
# implementation of the encoder network
class encoder(nn.Module):

  def __init__(self,seq_len, n_features, embedding_dim=64):
      super(encoder, self).__init__()

      self.seq_len, self.n_features = seq_len, n_features
      self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim


      self.encoder_L1 = nn.Linear(self.n_features, self.hidden_dim, bias=True)
      #nn.init.xavier_uniform_(self.encoder_L1.weight)
      self.encoder_R1 = nn.ReLU(True)


      self.encoder_L2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=True)
      #nn.init.xavier_uniform_(self.encoder_L2.weight)
      self.encoder_R2 = nn.ReLU(True )


      #self.dropout = nn.Dropout(p=0.0, inplace=True)

  def forward(self, x):

      x = self.encoder_L1(x)
      x = self.encoder_L2(x)

      return x
######################################################################
# implementation of the decoder network
class decoder(nn.Module):

  def __init__(self,seq_len, input_dim=64, n_features=1):
    super(decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.decoder_L1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)  # add linearity
    #nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [1]
    self.decoder_R1 = nn.ReLU(True )  # add non-linearity according to [2]

    self.decoder_L2 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_features, bias=True)  # add linearity
    #nn.init.xavier_uniform_(self.decoder_L2.weight)  # init weights according to [1]
    self.decoder_R2 = nn.ReLU(True)  # add non-linearity according to [2]

    # init dropout layer with probability p
    #self.dropout = nn.Dropout(p=0.0, inplace=True)

  def forward(self, x):
    # define forward pass through the network
    x = self.decoder_L1(x)
    x = self.decoder_L2(x)


    return x

##########################################################################
model_type="withLSTM"
#model_type="withoutLSTM"
class RecurrentAutoencoder(nn.Module):

      def __init__(self, seq_len, n_features, embedding_dim=64):
          super(RecurrentAutoencoder, self).__init__()

          self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
          self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)

          return x

########################################################################################################################
#model=Model()
#class Trainer(AgentMET4FOF):

    #     self.BenesBuffer = DataBuffer(buffer_length)
    #     DR.AllSensors[sensor_id].SetCallback(self.BenesBuffer.PushData)
#
    #     self.buffer_length = buffer_length
    #     self.buffer_length_realdata = buffer_length
    #     self.sensor_id = sensor_id
    #def __init__(self,number_columns_inbuffer,buffer_length):
    #def init_parameters(self):

       # self.buffer_length = 5#buffer_length
        #self.buffer_length_realdata = buffer_length



########################################################################################################################
    #def plot_real_data(self):
    #    start = timeit.timeit()
    #    while self.BenesBuffer.Datasetpushed < self.buffer_length:
    #        time.sleep(.001)
##
    #    df_test = DataFrame(columns=['unix_time', 'unix_time_nsecs', 'Data_01', 'Data_02', 'Data_03'])
#
    #    if not df_test.empty:
    #        del [[df_test]]
    #        gc.collect()
#
    #    for i in range(self.buffer_length):
    #        df_test.loc[i, 'unix_time'] = self.BenesBuffer.Buffer[i].unix_time
    #        df_test.loc[i, 'unix_time_nsecs'] = self.BenesBuffer.Buffer[i].unix_time_nsecs
    #        df_test.loc[i, 'Data_01'] = self.BenesBuffer.Buffer[i].Data_01
    #        df_test.loc[i, 'Data_02'] = self.BenesBuffer.Buffer[i].Data_02
    #        df_test.loc[i, 'Data_03'] = self.BenesBuffer.Buffer[i].Data_03
#
    #    df_test.index = (df_test['unix_time'] - self.BenesBuffer.firsttimestamp_s + df_test[
    #        'unix_time_nsecs'] / 1e9) - (
    #                            self.BenesBuffer.firsttimestamp_ns / 1e9)
    #    df_test = df_test[['Data_01', 'Data_02', 'Data_03']]
#
    #    test = df_test
    #    #x = test.iloc[:, [0]] * .005
    #    #y = test.iloc[:, [1]] * .1
    #    #z = test.iloc[:, [2]] * .01
    #    #test_sum = x.values + y.values + z.values
    #    #x = test.index
    #    #test.reset_index(drop=False, inplace=True)
#########################################################################################################################
class Predictor(AgentMET4FOF):
    def init_parameters(self):
        self.model = 0
        self.counter = 0
        self.var_Xtrain = 0

    def on_received_message(self,message):
        #start = time.time()
        #while self.BenesBuffer.Datasetpushed < self.buffer_length:
        #      time.sleep(.001)
#
        #df_test = DataFrame(columns=['unix_time', 'unix_time_nsecs', 'Data_01', 'Data_02', 'Data_03'])
        #df_test = pd.DataFrame(columns=["time", "x"])
#


        #if self.pointer < 5:
        #    self.data = pd.DataFrame.from_dict(self.ssupertream).iloc[self.pointer:self.pointer+1]
        #    self.pointer += 1


        #if not df_test.empty :
        #   del [[df_test]]
        #   gc.collect()
##
        #for i in range(self.buffer_length):
        #    df_test.loc[i, 'unix_time'] = self.BenesBuffer.Buffer[i].unix_time
        #    df_test.loc[i, 'unix_time_nsecs'] = self.BenesBuffer.Buffer[i].unix_time_nsecs
        #    df_test.loc[i, 'Data_01'] = self.BenesBuffer.Buffer[i].Data_01
        #    df_test.loc[i, 'Data_02'] = self.BenesBuffer.Buffer[i].Data_02
        #    df_test.loc[i, 'Data_03'] = self.BenesBuffer.Buffer[i].Data_03
##
        #df_test.index = (df_test['unix_time'] - self.BenesBuffer.firsttimestamp_s + df_test['unix_time_nsecs'] / 1e9) - (
        #        self.BenesBuffer.firsttimestamp_ns / 1e9)
        #df_test=df_test[['Data_01','Data_02','Data_03']]
###########################################################################################################################

        #df_test.index = df_test["time"]
        #df_test = df_test.drop("time", axis=1)
###########################################################################################################################
        # handle Trainer Agent
        #self.model = 0

        self.best_loss = 10000.0
        self.n_epochs = 5


        self.log_info("trainer_Start")
        # x = input()
        # print('Hello, ' + x)

        X_train = pd.DataFrame(message["data"]['y'])
        print(f'X_train:{X_train}')


        self.counter +=len(X_train)
        print(f'input:{X_train}')
        if self.counter<=len(X_train):

           self.var_Xtrain = np.array(X_train)
           self.var_Xtrain = self.var_Xtrain.var(axis=0)
           print(f'var_Xtrain:{self.var_Xtrain}')

           print(f'ctr:{self.counter}')
           X_train, seq_len, n_features = create_dataset(X_train)

           self.model = RecurrentAutoencoder(seq_len, n_features, 64)
           self.model = self.model.to(device)

           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
           self.criterion = nn.L1Loss(reduction='sum').to(device)
           self.history = dict(train=[], val=[])

           self.best_model_wts = copy.deepcopy(self.model.state_dict())

           for epoch in range(1, self.n_epochs + 1):
               self.model = self.model.train()

               train_losses = []
               for seq_true in X_train:
                   self.optimizer.zero_grad()

                   seq_true = seq_true.to(device)
                   seq_pred = self.model(seq_true)

                   loss = self.criterion(seq_pred, seq_true)

                   loss.backward()
                   self.optimizer.step()

                   train_losses.append(loss.item())

               train_loss = np.mean(train_losses)

               self.history['train'].append(train_loss)

               # if val_loss < best_loss:
               #  best_loss = val_loss
               #  best_model_wts = copy.deepcopy(model.state_dict())

               print(f'Epoch:{model_type} {epoch}: train loss {np.round(train_loss, 3)}')  # val loss {np.round(val_loss)}')

               self.model.load_state_dict(self.best_model_wts)
           # return model.eval(), self.history
           self.log_info("trainer_End")
           #self.send_output({'model': self.model.eval()})
########################################################################################################################

        elif self.counter>len(X_train):
                print(f'cts:{self.counter}')
                df_test=0
                now = datetime.now()
                df_test = message["data"]['y']

                # if message["from"]=="AnomaliesGeneratorAgent_Test" and now.second % 5 != 0:
                if  now.second % 5 != 0:
                    r = np.random.uniform(-2, 2, size=1)
                    df_test = message["data"]['y']*r
                    print(f'inputan:{df_test}')
               # elif message["from"]=="SineGeneratorAgent_Test1" and now.second % 5 != 0:
                #    self.log_info("ttt")
                 #   df_test = message["data"]['y']

                df_test = pd.DataFrame(df_test)
                self.log_info("df_test_begin")
                df_test, seq_len, n_features = create_dataset(df_test)

                self.log_info("df_test_finish")

                predictions, losses = [], []
                #criterion = nn.L1Loss(reduction='sum').to(device)
                with torch.no_grad():
                    self.model = self.model.eval()
                    for seq_true in df_test:
                        seq_true = seq_true.to(device)
                        seq_pred = self.model(seq_true)

                        #loss = criterion(seq_pred, seq_true)

                        predictions.append(seq_pred.cpu().numpy().flatten())
                        #losses.append(loss.item())

                df_test = [t.tolist() for t in df_test]
                print(f'df_test: {df_test}')
                df_test=pd.DataFrame(np.squeeze(df_test, axis=1))
                print(f'df_test2: {df_test}')
                X_test = np.array(df_test)
                #X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                #X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                X_pred = [t.tolist() for t in predictions]
                X_pred = pd.DataFrame(np.squeeze(X_pred, axis=1))
                Xpred = pd.DataFrame(X_pred)
                #Xpred=np.array(X_pred)
                #X_pred.index = message["data"]['time']
############################################################################################################################
                var_Xtest = 0
                loss = 0
                uncertainty_loss_der_square = 0
                uncertainty_loss = 0
                z_scores = 0
                p_values = 0
                scored = pd.DataFrame()
                test_sum = pd.DataFrame()
                #uncertainty_loss_der_square=pd.DataFrame()
                #Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
                Xtest =X_test
                if not scored.empty and test_sum.empty :#and uncertainty_loss_der_square:
                  del [[scored, test_sum]]#,uncertainty_loss_der_square]]
                  gc.collect()

                x = 0
                y = 0
                z = 0
                print(f'Xtest: {Xtest}')
                var_Xtest = Xtest.var(axis=0)
                print(f'var_Xtest: {var_Xtest}')
                #loss = (1 / 3) * np.sum(((Xtest - X_pred) ** 2), axis=1)
                loss =  np.sum(((Xtest - X_pred) ** 2), axis=1)
                print(f'mse: {loss}')
                l1 = []
                l2 = []

                for i in range(1):
                    l1 = ((Xtest[:, i] - Xpred.iloc[:, i]) / 2) ** 2
                    l2.append(l1)

                uncertainty_loss_der_square = pd.DataFrame(np.transpose(l2))
                print(f'uncertainty_loss_der_square: {uncertainty_loss_der_square}')
                print(f'var_Xtest: {var_Xtest}')
                uncertainty_loss = np.sum(uncertainty_loss_der_square * var_Xtest, axis=1)
                print(f'uncertainty_loss: {uncertainty_loss}')

            #threshold = 74.315067 #* 3
                threshold =self.var_Xtrain[0]*3
                print(f'threshold: {threshold}')

                z_scores = (threshold - loss) / np.sqrt(uncertainty_loss.values)
                print(f'z_scores: {z_scores}')
                p_values =stats.norm.cdf(z_scores)
                print(f'p_values: {p_values}')

                scored['loss'] = loss
                scored['threshold'] = threshold
                #scored['uncertainty_loss'] = uncertainty_loss.values
                scored['upper_uncertainty_loss'] = loss + uncertainty_loss.values
                scored['below_uncertainty_loss'] = loss - uncertainty_loss.values
                scored['p_values'] = 1-p_values

        #test_sum = test.index
#
        #x = test.iloc[:, [0]] * .005
        #y = test.iloc[:, [1]] * .1
        #z = test.iloc[:, [2]] * .01
        #test_sum = x.values + y.values + z.values
        #test['test_sum'] = test_sum#np.abs(test_sum)

       #if scored.iloc[:, 0].mean()>threshold:
       #    df_anomalies.loc[scored.index[-1],'loss']=scored.iloc[:, 0].mean()
       #    df_anomalies.loc[scored.index[-1], 'dif_threshold'] = scored.iloc[:, 0].mean()-threshold
            #df_anomalies.to_csv('Results_Anomalies/'+now.strftime("%Y-%m-%d___%H_%M_%S")+'.csv', sep=';', mode='w')

        #end = time.time()
        #p = end - start
        #u=np.round(p,3)
        #print(u)
                self.scored_dict=scored.to_dict("list")
                self.send_output(self.scored_dict)
        else:
                self.log_info("train_model not available!!!")

#############################################################################################################################
def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents

    gen_agent_test1 = agentNetwork.add_agent(agentType=SineGeneratorAgent_Test1, log_mode=False)
    #gen_agent_test2 = agentNetwork.add_agent(agentType=SineGeneratorAgent_Test2, log_mode=False)
    #gen_agent_test3 = agentNetwork.add_agent(agentType=SineGeneratorAgent_Test3, log_mode=False)



    predictor_agent = agentNetwork.add_agent(agentType=Predictor)

    monitor_agent_1 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=1000,log_mode=False)
    monitor_agent_2 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=1000,log_mode=False)


    gen_agent_test1.init_parameters(sensor_buffer_size=10)
    gen_agent_test1.init_agent_loop(loop_wait=.1)
#
    #trainer_agent.init_parameters()
    predictor_agent.init_parameters()
    ##gen_agent.init_parameters(stream=SineGenerator(), pretrain_size=1000,batch_size=1)

    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple input types
    ##agentNetwork.bind_agents(gen_agent, trainer_agent)
    # This monitor agent will only store 'x' of the data keys into its memory

    monitor_agent_1.init_parameters(plot_filter=['y'])

    #agentNetwork.bind_agents(gen_agent_train, trainer_agent)
    agentNetwork.bind_agents(gen_agent_test1, predictor_agent)
    #agentNetwork.bind_agents(gen_agent_anomaly, predictor_agent)

    #agentNetwork.bind_agents(trainer_agent, predictor_agent)
    agentNetwork.bind_agents(gen_agent_test1, monitor_agent_1)
    #agentNetwork.bind_agents(gen_agent_test2, monitor_agent_1)
    #agentNetwork.bind_agents(gen_agent_test3, monitor_agent_1)

    agentNetwork.bind_agents(predictor_agent, monitor_agent_2)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()