import gc
#from MET4FOFDataReceiver import DataBuffer,DR
import time
#import psutil
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from datetime import datetime
import os

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
from examples import custom_dashboard

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
        def init_parameters(self, sensor_buffer_size,c1=.3,c2=.6):
            self.stream = SineGenerator()
            self.buffer_size = sensor_buffer_size
            self.c1=c1
            self.c2=c2
        def agent_loop(self):
            if self.current_state == "Running":
                sine_data = self.stream.next_sample()  # dictionary
                #current_time = datetime.now().second
                current_time = datetime.today().strftime("%H:%M:%S.%f")[:-3]
                sine_data = {'Time':current_time,'y1': sine_data['x'],'y2': sine_data['x']*self.c1,'y3': sine_data['x']*self.c2}#'Time':current_time,

                # save data into memory
                # self.name="SineGeneratorAgent_Test1"
                self.update_data_memory({'from': self.name, 'data': sine_data})
                # send out buffered data if the stored data has exceeded the buffer size
                if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                    print(f'www:{self.memory[self.name]}')
                    self.send_output(self.memory[self.name])
                    self.memory = {}


# class SineGeneratorAgent_Test2(AgentMET4FOF):
#     def init_parameters(self, sensor_buffer_size):
#         self.stream = SineGenerator()
#         self.buffer_size = sensor_buffer_size
#
#     def agent_loop(self):
#         if self.current_state == "Running":
#             sine_data = self.stream.next_sample()  # dictionary
#             now = datetime.now()
#             current_time = datetime.today().strftime("%H:%M:%S")
#             sine_data = {'Time':current_time,'y2': sine_data['x']}#
#
#             # save data into memory
#             # self.name="SineGeneratorAgent_Test1"
#             self.update_data_memory({'from': self.name, 'data': sine_data})
#             # send out buffered data if the stored data has exceeded the buffer size
#             if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
#                 self.send_output(self.memory[self.name])
#                 self.memory = {}
#
# class SineGeneratorAgent_Test3(AgentMET4FOF):
#         def init_parameters(self, sensor_buffer_size):
#             self.stream = SineGenerator()
#             self.buffer_size = sensor_buffer_size
#
#         def agent_loop(self):
#             if self.current_state == "Running":
#                 sine_data = self.stream.next_sample()  # dictionary
#                 now = datetime.now()
#                 current_time = datetime.today().strftime("%H:%M:%S")
#                 sine_data = {'Time':current_time,'y3': sine_data['x']}#
#
#                 # save data into memory
#                 # self.name="SineGeneratorAgent_Test1"
#                 self.update_data_memory({'from': self.name, 'data': sine_data})
#                 # send out buffered data if the stored data has exceeded the buffer size
#                 if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
#                     self.send_output(self.memory[self.name])
#                     self.memory = {}

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
#model_type="withLSTM"
model_type="withoutLSTM"
class RecurrentAutoencoder(nn.Module):

      def __init__(self, seq_len, n_features, embedding_dim=64):
          super(RecurrentAutoencoder, self).__init__()

          self.encoder = encoder(seq_len, n_features, embedding_dim).to(device)
          self.decoder = decoder(seq_len, embedding_dim, n_features).to(device)

      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)

          return x

########################################################################################################################
class Aggregator(AgentMET4FOF):
    def init_parameters(self):
        self.df =[]
    def on_received_message(self, message):
        self.update_data_memory(message)
        print("majid")
        self.log_info("majid:"+str(self.memory))
        self.df.append(self.memory)
        print(f'df:{self.df}')
        #
        self.send_output(self.df)


class Predictor(AgentMET4FOF):
    def init_parameters(self,train_size):
        self.model = 0
        self.counter = 0
        self.X_train_mean = 0
        self.X_train_std = 0
        self.n_epochs = 5
        self.train_size=train_size
        self.X_train_df = pd.DataFrame()

        # #snippet
        # if self.counter == self.trainer_counter:
        #     #train model
        #     self.train_model()
        # elif self.model != 0: #model is not none
        #     self.model.predict()
    def on_received_message(self,message):
        self.best_loss = 10000.0
        self.log_info("trainer_Start")

        # self.X_train_df = pd.DataFrame([message["data"]['y1'], message["data"]['y2'], message["data"]['y3']])
        # print(f'X_train_df:{self.X_train_df}')

        # self.counter +=len(self.X_train_df)

        # if self.counter<=len(self.X_train_df):
        X_train_df_temp = pd.DataFrame([message["data"]['y1'], message["data"]['y2'], message["data"]['y3']])
        X_train_df_temp = X_train_df_temp.T
        self.counter += len(X_train_df_temp)
#########################
        if self.counter <= self.train_size:

           self.X_train_df=self.X_train_df.append(X_train_df_temp)
           print(f'X_train_df:{self.X_train_df}')
#################
        if self.counter>self.train_size and self.model == 0:
           #self.X_train_df = self.X_train_df.reset_index()

           print(f'full_X_train_df:{self.X_train_df}')

           #self.X_train_mean = X_train_df.mean(axis=0).values.reshape(1, -1)

           self.X_train_std = self.X_train_df.std(axis=0)
           #self.Xtrain = np.array(X_train)
           #self.std_Xtrain = self.Xtrain.std(axis=0)
           print(f'X_train_std:{self.X_train_std}')

           print(f'ctr:{self.counter}')
           X_train_df, seq_len, n_features = create_dataset(self.X_train_df)

           self.model = RecurrentAutoencoder(seq_len, n_features, 64)
           self.model = self.model.to(device)

           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
           self.criterion = nn.L1Loss(reduction='sum').to(device)
           self.history = dict(train=[], val=[])

           self.best_model_wts = copy.deepcopy(self.model.state_dict())

           for epoch in range(1, self.n_epochs + 1):
               self.model = self.model.train()

               train_losses = []
               for seq_true in X_train_df:
                   self.optimizer.zero_grad()

                   seq_true = seq_true.to(device)
                   seq_pred = self.model(seq_true)

                   loss = self.criterion(seq_pred, seq_true)

                   loss.backward()
                   self.optimizer.step()

                   train_losses.append(loss.item())

               train_loss = np.mean(train_losses)

               self.history['train'].append(train_loss)

               print(f'Epoch:{model_type} {epoch}: train loss {np.round(train_loss, 3)}')  # val loss {np.round(val_loss)}')

               self.model.load_state_dict(self.best_model_wts)
           # return model.eval(), self.history
           self.log_info("trainer_End")
           #self.send_output({'model': self.model.eval()})
########################################################################################################################

        #elif self.counter>len(X_train_df):
        elif self.model != 0 :
                print(f'cts:{self.counter}')
                X_test_df=0
                now = datetime.now()
                X_test_df =pd.DataFrame([message["data"]['y1'],message["data"]['y2'],message["data"]['y3']])
                X_test_df=X_test_df.T
                print(f'X_test_df:{X_test_df}')
                # if message["from"]=="AnomaliesGeneratorAgent_Test" and now.second % 5 != 0:
                if  now.second % 5 != 0:
                    r = np.random.uniform(-2,2 , size=1)
                    print(f'r:{r}')
                    X_test_df = pd.DataFrame([message["data"]['y1']*r,message["data"]['y2']*r*.5,message["data"]['y3']*r*1.5])
                    X_test_df = X_test_df.T
                    print(f'abnormal_test:{X_test_df}')
               # elif message["from"]=="SineGeneratorAgent_Test1" and now.second % 5 != 0:
                #    self.log_info("ttt")
                 #   df_test = message["data"]['y']
                if now.second % 5 == 0:
                    X_test_df = pd.DataFrame([message["data"]['y1'],message["data"]['y2'],message["data"]['y3']])
                    X_test_df = X_test_df.T
                    print(f'normal_test:{X_test_df}')
                self.log_info("X_test_df_begin")
                #X_test_mean = X_train_df.mean(axis=0).values.reshape(1, -1)
                X_test_df, seq_len, n_features = create_dataset(X_test_df)

                self.log_info("X_test_df_finish")

                predictions, losses = [], []
                #criterion = nn.L1Loss(reduction='sum').to(device)
                with torch.no_grad():
                    self.model = self.model.eval()
                    for seq_true in X_test_df:
                        seq_true = seq_true.to(device)
                        seq_pred = self.model(seq_true)

                        #loss = criterion(seq_pred, seq_true)

                        predictions.append(seq_pred.cpu().numpy().flatten())
                        #losses.append(loss.item())

                X_test_df = [t.tolist() for t in X_test_df]
                print(f'X_test_df: {X_test_df}')
                X_test_df=pd.DataFrame(np.squeeze(X_test_df))
                print(f'X_test_df2: {X_test_df}')
                X_test_arr = np.array(X_test_df)
                print(f'X_test_arr: {X_test_arr}')
                #X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                #X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                X_pred = [t.tolist() for t in predictions]
                X_pred_df = pd.DataFrame(np.squeeze(X_pred))
                print(f'X_pred_df: {X_pred_df}')
                Xpred_arr = np.array(X_pred_df)
                print(f'Xpred_arr: {Xpred_arr}')
                #Xpred=np.array(X_pred)
                #X_pred.index = message["data"]['time']
############################################################################################################################
                var_Xtest = 0
                std_Xtest=0
                loss = 0
                uncertainty_loss_der_square = 0
                uncertainty_loss = 0
                z_scores = 0
                p_values = 0
                scored = pd.DataFrame()
                test_sum = pd.DataFrame()
                #uncertainty_loss_der_square=pd.DataFrame()
                #Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])

                if not scored.empty and test_sum.empty :#and uncertainty_loss_der_square:
                  del [[scored, test_sum]]#,uncertainty_loss_der_square]]
                  gc.collect()

                x = 0
                y = 0
                z = 0
                print(f'Xtest: {X_test_arr}')
                var_Xtest = X_test_arr.var(axis=0)
                std_Xtest = X_test_arr.std(axis=0)
                print(f'var_Xtest: {var_Xtest}')
                #loss = (1 / 3) * np.sum(((Xtest - X_pred) ** 2), axis=1)
                loss =  (1/len(X_test_df.columns))*np.sum(((X_test_arr - Xpred_arr) ** 2), axis=0)
                print(f'loss: {loss}')
                print(f'mse: {loss}')
                l1 = []
                l2 = []

                for i in range(len(X_test_df.columns)):
                    l1 = ((X_test_arr[:, i] - X_pred_df.iloc[:, i]) / 2) ** 2
                    l2.append(l1)

                uncertainty_loss_der_square = pd.DataFrame(np.transpose(l2))
                print(f'uncertainty_loss_der_square: {uncertainty_loss_der_square}')

                uncertainty_loss = np.sum(uncertainty_loss_der_square * var_Xtest, axis=0)
                print(f'uncertainty_loss: {uncertainty_loss}')

            #threshold = 74.315067 #* 3
                print(f'X_train_std: {self.X_train_std}')
                threshold =[np.mean(self.X_train_std)*2]# 95.4% confidence interval
                print(f'threshold: {threshold}')

                z_scores = (threshold - loss) / np.sqrt(uncertainty_loss.values)
                print(f'z_scores: {z_scores}')
                p_values =stats.norm.cdf(z_scores)
                print(f'p_values: {p_values}')

                scored['loss'] =[np.mean(loss)]

                scored['threshold'] =threshold
                print("threshold:",scored['threshold'])#threshold
                #scored['uncertainty_loss'] = uncertainty_loss.values
                scored['upper_uncertainty_loss'] = np.mean([loss + uncertainty_loss.values])
                scored['below_uncertainty_loss'] = np.mean([loss - uncertainty_loss.values])
                scored['p_values'] = np.mean([1-p_values])
                scored['Time'] =message['data']['Time'][0]
                #print(f"ttt:{message['data']['Time'][0]}")
                print(f'scored:{scored.T}')

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
                print(f'scored_dict:{self.scored_dict}')
                self.send_output(self.scored_dict)
        else:
                self.log_info("train_model not available!!!")
def custom_create_monitor_graph1(data, sender_agent):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters
    """

    x=data['Time']
    #x= datetime.now().second
    y1 = data['y1']
    y2 = data['y2']
    y3 = data['y3']

    all_go = [go.Scatter(x=x, y=y1, mode="lines", name='Sensor1',line=dict(color="red")),
              go.Scatter(x=x, y=y2, mode="lines", name='Sensor2',line=dict(color="green")),
              go.Scatter(x=x, y=y3, mode="lines", name='Sensor3',line=dict(color="blue"))]
    return all_go

def custom_create_monitor_graph2(data, sender_agent):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters
    """

    x=data['Time']
    #x= datetime.now().second
    loss = data['loss']
    threshold = data['threshold']
    upper_uncertainty_loss = data['upper_uncertainty_loss']
    below_uncertainty_loss = data['below_uncertainty_loss']
    p_values = data['p_values']

    all_go = [go.Scatter(x=x, y=loss, mode="lines", name='loss',line=dict(color="#3399FF")),
              go.Scatter(x=x, y=threshold, mode="lines", name='threshold',line=dict(color="yellow")),
              go.Scatter(x=x, y=upper_uncertainty_loss, mode="lines", name='upper_uncertainty_loss',line=dict(color="#CCE5FF")),
              go.Scatter(x=x, y=below_uncertainty_loss, mode="lines", name='below_uncertainty_loss',line=dict(color="#CCE5FF")),
              go.Scatter(x=x, y=p_values, mode="lines", name='p_values',line=dict(color="#FF66B2"))]
    return all_go
#############################################################################################################################
def main():
    # start agent network server
    agentNetwork = AgentNetwork(dashboard_update_interval=.5)

    gen_agent_test1 = agentNetwork.add_agent(name="Sensor1", agentType=SineGeneratorAgent_Test1, log_mode=False)
    #gen_agent_test2 = agentNetwork.add_agent(name="Sensor2",agentType=SineGeneratorAgent_Test2, log_mode=False)
    #gen_agent_test3 = agentNetwork.add_agent(name="Sensor3",agentType=SineGeneratorAgent_Test3, log_mode=False)

    # predictor_agent = agentNetwork.add_agent(agentType=Predictor)
    #aggregator_agent = agentNetwork.add_agent(agentType=Aggregator)
    predictor_agent = agentNetwork.add_agent(agentType=Predictor)
    monitor_agent_1 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=100,log_mode=False)
    monitor_agent_2 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=100,log_mode=False)

    #setting agent parameters
    gen_agent_test1.init_parameters(sensor_buffer_size=1)
    #gen_agent_test2.init_agent_loop(loop_wait=.1)
    #gen_agent_test3.init_parameters(sensor_buffer_size=5)
    gen_agent_test1.init_agent_loop(loop_wait=.1)
    #gen_agent_test2.init_parameters(sensor_buffer_size=5)
    #gen_agent_test3.init_agent_loop(loop_wait=.1)

    #aggregator_agent.init_parameters()
    predictor_agent.init_parameters(100)#define train_size
    monitor_agent_1.init_parameters(plot_filter=['Time','y1','y2','y3'],custom_plot_function=custom_create_monitor_graph1)
    monitor_agent_2.init_parameters(plot_filter=['Time', 'loss', 'threshold', 'upper_uncertainty_loss','below_uncertainty_loss','p_values'],
                                    custom_plot_function=custom_create_monitor_graph2)


    #bind agents
    #agentNetwork.bind_agents(gen_agent_test1, aggregator_agent)
    #agentNetwork.bind_agents(gen_agent_test2, aggregator_agent)
    #agentNetwork.bind_agents(gen_agent_test3, aggregator_agent)
    agentNetwork.bind_agents(gen_agent_test1, predictor_agent)

    #agentNetwork.bind_agents(Aggregator_agent, predictor_agent)
    #agentNetwork.bind_agents(Aggregator_agent, monitor_agent_1)
    agentNetwork.bind_agents(gen_agent_test1, monitor_agent_1)
    agentNetwork.bind_agents(predictor_agent, monitor_agent_2)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()