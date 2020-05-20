import gc
import time

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
from examples import custom_dashboard
#######################################################################################################################
#Defined random_seed because we want same raw data in every runing
random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

#Defenition of process engine of pytorch model, so if some machine does not have GPU we could use CPU instead
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#Generating raw data
class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self, sensor_buffer_size, scale_amplitude=1):
        self.stream = SineGenerator()
        self.buffer_size = sensor_buffer_size
        self.scale_amplitude = scale_amplitude

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample()
            #current_time = datetime.today().strftime("%H:%M:%S.%f")[:-3]
            # current_time = current_time.split()[0].replace(':', '')
            # current_time = float(current_time)
            current_time = time.time()

            if self.scale_amplitude==1:
                sine_data = {'Time': current_time, 'y1': sine_data['x'] * self.scale_amplitude}
            elif self.scale_amplitude==.3:
                sine_data = {'Time': current_time, 'y2': sine_data['x'] * self.scale_amplitude}
            elif self.scale_amplitude==.6:
                sine_data = {'Time': current_time, 'y3': sine_data['x'] * self.scale_amplitude}

            self.update_data_memory({'from': self.name, 'data': sine_data})
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

########################################################################################################################
#Auto encoder consisted by two classes Encoder and Decoder
#We have two types of Encoder and Decoder that names withLSTM and withoutLSTM
#Definition of three important variables
'''
seq_len means number of columns
n_seq means number of rows
n_features means each tensor that is one

'''

# Each of Encoder_withLSTM and Decoder_withLSTM has two layers of LSTM that call rnn1 and rnn2
class Encoder_withLSTM(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder_withLSTM, self).__init__()

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
class Decoder_withLSTM(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder_withLSTM, self).__init__()

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
# Each of Encoder_withoutLSTM and Decoder_withoutLSTM has two layers that call L1 and L2
class Encoder_withoutLSTM(nn.Module):

  def __init__(self,seq_len, n_features, embedding_dim=64):
      super(Encoder_withoutLSTM, self).__init__()

      self.seq_len, self.n_features = seq_len, n_features
      self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim


      self.encoder_L1 = nn.Linear(self.n_features, self.hidden_dim, bias=True)
      self.encoder_R1 = nn.ReLU(True)


      self.encoder_L2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=True)
      self.encoder_R2 = nn.ReLU(True )


  def forward(self, x):

      x = self.encoder_L1(x)
      x = self.encoder_L2(x)

      return x
######################################################################
# implementation of the decoder network
class Decoder_withoutLSTM(nn.Module):

  def __init__(self,seq_len, input_dim=64, n_features=1):
    super(Decoder_withoutLSTM, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.decoder_L1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)  # add linearity
    self.decoder_R1 = nn.ReLU(True )  # add non-linearity according to [2]

    self.decoder_L2 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_features, bias=True)  # add linearity
    self.decoder_R2 = nn.ReLU(True)  # add non-linearity according to [2]

  def forward(self, x):
    # define forward pass through the network
    x = self.decoder_L1(x)
    x = self.decoder_L2(x)

    return x

##########################################################################

class RecurrentAutoencoder(nn.Module):

      def __init__(self, seq_len, n_features, embedding_dim=64,model_type="withoutLSTM"):
          super(RecurrentAutoencoder, self).__init__()

          if model_type=="withoutLSTM":
            self.encoder = Encoder_withoutLSTM(seq_len, n_features, embedding_dim).to(device)
            self.decoder = Decoder_withoutLSTM(seq_len, embedding_dim, n_features).to(device)

          elif model_type=="withLSTM":
              self.encoder = Encoder_withLSTM(seq_len, n_features, embedding_dim).to(device)
              self.decoder = Decoder_withLSTM(seq_len, embedding_dim, n_features).to(device)

      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)

          return x

########################################################################################################################
#Aggregator class aggregate three different raw sensors data to one output sensor data
class Aggregator(AgentMET4FOF):

    def on_received_message(self, message):
        self.update_data_memory(message)
        print(f'message:{message}')
        self.log_info("self.memory:"+str(self.memory))

        if 'Sensor1_1' in self.memory and 'Sensor2_1' in self.memory and 'Sensor3_1' in self.memory:
            a = pd.DataFrame(self.memory['Sensor1_1'])
            b = pd.DataFrame(self.memory['Sensor2_1'])
            c = pd.DataFrame(self.memory['Sensor3_1'])

            agg_df = pd.concat([a, b, c], axis=1)
            agg_df = agg_df.loc[:, ~agg_df.columns.duplicated()]

            self.memory={}
            print(f'agg_df:{agg_df}')
            self.send_output(agg_df.to_dict("list"))

#Predictor class is the most important class that we train, predict and calculate loss, uncertainties, threshold and p_value
class Predictor(AgentMET4FOF):
    def init_parameters(self,train_size,model_type):
        self.model = None
        self.counter = 0
        self.X_train_mean = 0
        self.X_train_std = 0
        self.n_epochs = 5
        self.train_size=train_size
        self.X_train_df = pd.DataFrame()
        self.model_type=model_type
        self.path=str(Path(__file__).parent)+"/saved_trained_models/"
        self.train_loss_best=[]
        self.best_epoch=None
        self.best_model_wts=None

#Function create_dataset receive a dataframe parameter and convert it to list of tensors that pytorch model could read it
    def create_dataset(self,df):
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features

    def on_received_message(self,message):

        self.log_info("trainer_Start")

        print(f'agg_message:{message}')
        X_train_df_temp = pd.DataFrame([message['data']['y1'], message['data']['y2'], message['data']['y3']])
        X_train_df_temp = X_train_df_temp.T
        print(f'X_train_df_temp:{X_train_df_temp}')

#In below if block we fill a X_train_df in amount of train_size by considering in every loop  that
# we receive X_train_df_temp that it's size is amount of buffer size
        if self.counter < self.train_size:
           self.counter += len(X_train_df_temp)

           self.X_train_df=self.X_train_df.append(X_train_df_temp.head(self.train_size))
           self.X_train_df=self.X_train_df.reset_index(drop=True)
           print(f'X_train_df:{self.X_train_df}')

#In below if block we X_train_df filled in amount of train_size, so we start to train model with X_train_df
#self.counter>=self.train_size means X_train_df filled in amount of train_size
#self.model == None means our model did not tarin yet and is empty
        if self.counter>=self.train_size and self.model == None:
           print(f'counter1:{self.counter}')

           print(f'full_X_train_df:{self.X_train_df}')

           self.X_train_std = self.X_train_df.std(axis=0)

           print(f'X_train_std:{self.X_train_std}')


           X_train_df, seq_len, n_features =self.create_dataset(self.X_train_df)

           print(f'seq_len:{seq_len}')
           print(f'n_features:{n_features}')

#Below block initialize train model with defining pytorch model hyperparameters
           self.model = RecurrentAutoencoder(seq_len, n_features, 64,self.model_type)
           self.model = self.model.to(device)

           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
           self.criterion = nn.L1Loss(reduction='sum').to(device)

           self.best_model_wts = copy.deepcopy(self.model.state_dict())
           print(f'state_dict:{self.model.state_dict()}')

#In below for block we train model with some number of epochs
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
               self.train_loss_best.append(train_loss)
               print(f'self.model.state_dict():{self.model.state_dict()}')

#In below if block we put state_dict of best model(means each model that has minimum loss) in best_model_wts variable
#In addition, train_loss_best[0] has minimum loss value
               if train_loss<self.train_loss_best[0]:
                  self.train_loss_best[0]=train_loss

                  self.best_model_wts = self.model.state_dict()
                  print(f'best_model_wts:{self.best_model_wts}')
                  self.best_epoch=epoch

               print(f'Epoch:{self.model_type} {epoch}: train loss {np.round(train_loss, 4)}')

           now = datetime.now()
#In below block we save best_model_wts in hard disk
           torch.save(self.best_model_wts,
                      self.path + now.strftime("%Y-%m-%d___%H_%M_%S")+ '_train_epoch{}.pth'.format(self.best_epoch))

           print(f'train_loss_best:{np.round(self.train_loss_best, 4)}')
           print(f'best_model_wts:{self.best_model_wts}')

           self.log_info("trainer_End")
########################################################################################################################
# self.model == None means our model tarined and is ready to predict new input streams raw data
        elif self.model != None :
                print(f'counter2:{self.counter}')
                X_test_df=0
                now = datetime.now()

                print(f'message_test:{message}')

#In below if block we generate every 5 seconds random anomaly data and multiple to each sensor data, random anomaly data mulipuled with fixed decimal number because we want different anomalies for each sensor
                if  now.second % 5 == 0:
                    random_anomaly = np.random.uniform(-2,2 , size=1)
                    print(f'r:{random_anomaly}')
                    X_test_df = pd.DataFrame([message['data']['y1']*random_anomaly, message['data']['y2']*random_anomaly*.3, message['data']['y3']*random_anomaly*.6])
                    X_test_df = X_test_df.T
                    print(f'abnormal_test:{X_test_df}')

# In below if block we generate normal data during every 5 seconds
                if now.second % 5 != 0:
                    X_test_df = pd.DataFrame([message['data']['y1'], message['data']['y2'], message['data']['y3']])
                    X_test_df = X_test_df.T
                    print(f'normal_test:{X_test_df}')
                self.log_info("X_test_df_begin:")

                X_test_torch, seq_len, n_features = self.create_dataset(X_test_df)

# In below block we pass every stream blocks of data(in amount of buffer size) to model for prediction
                predictions, losses = [], []
                with torch.no_grad():
                    self.model = self.model.eval()
                    for seq_true in X_test_torch:
                        seq_true = seq_true.to(device)
                        seq_pred = self.model(seq_true)

                        predictions.append(seq_pred.cpu().numpy().flatten())

                X_test_torch = [t.tolist() for t in X_test_torch]
                print(f'X_test_torch_3dim: {X_test_torch}')

#Here convert 3dim list data to pandas dataframe
                X_test_df=pd.DataFrame(np.squeeze(X_test_torch))
                print(f'X_test_df: {X_test_df}')

#Here convert pandas dataframe of np.array for better and easier next calculations
                X_test_arr = np.array(X_test_df)
                print(f'X_test_arr: {X_test_arr}')

#Here convert 3dim list prediction data to pandas dataframe
                X_pred = [t.tolist() for t in predictions]
                X_pred_df = pd.DataFrame(np.squeeze(X_pred))
                print(f'X_pred_df: {X_pred_df}')

#Here convert pandas dataframe of np.array for better and easier next calculations
                Xpred_arr = np.array(X_pred_df)
                print(f'Xpred_arr: {Xpred_arr}')

############################################################################################################################
                var_Xtest = 0
                loss = 0
                uncertainty_loss_der_square = 0
                uncertainty_loss = 0
                z_scores = 0
                p_values = 0

#scored data frame consisted by all calculation's results
                scored = pd.DataFrame()

#In below if block we delete scored dataframe after each iteration for preventing overflow in memory
                if not scored.empty:
                  del [scored]
                  gc.collect()

                var_Xtest = X_test_arr.var(axis=0)
                print(f'var_Xtest: {var_Xtest}')

#Here we calculate loss value we used mean square error formula for calculating of loss
                loss =  (1/len(X_test_df.columns))*np.sum(((X_test_arr - Xpred_arr) ** 2), axis=0)
                print(f'MSE: {loss}')

#Here we calculate derivative of loss for using in standard uncertainty
                l1 = []
                l2 = []

                for i in range(len(X_test_df.columns)):
                    l1 = ((X_test_arr[:, i] - X_pred_df.iloc[:, i]) / 2) ** 2
                    l2.append(l1)

                uncertainty_loss_der_square = pd.DataFrame(np.transpose(l2))
                print(f'uncertainty_loss_der_square: {uncertainty_loss_der_square}')

#Here we calculate standard uncertainty
                uncertainty_loss = np.sum(uncertainty_loss_der_square * var_Xtest, axis=0)
                print(f'uncertainty_loss: {uncertainty_loss}')

#Here we define threshold that is calculated by mean of standard deviation of train data of three sensors
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
                scored['upper_uncertainty_loss'] = np.mean([loss + uncertainty_loss.values])
                scored['below_uncertainty_loss'] = np.mean([loss - uncertainty_loss.values])

                scored['p_values'] = np.mean([1-p_values])
                scored['Time'] =message['data']['Time'][0]
                print(f'scored:{scored.T}')

                self.scored_dict=scored.to_dict("list")
                print(f'scored_dict:{self.scored_dict}')
                self.send_output(self.scored_dict)
        else:
                self.log_info("train_model not available!!!")

def custom_create_monitor_graph_actualdata(data, sender_agent):
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
    y1 = data['y1']
    y2 = data['y2']
    y3 = data['y3']

    all_go = [go.Scatter(x=x, y=y1, mode="lines", name='Sensor1',line=dict(color="red")),
              go.Scatter(x=x, y=y2, mode="lines", name='Sensor2',line=dict(color="green")),
              go.Scatter(x=x, y=y3, mode="lines", name='Sensor3',line=dict(color="blue"))]
    return all_go

def custom_create_monitor_graph_calculation(data, sender_agent):
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
    loss = data['loss']
    threshold = data['threshold']
    upper_uncertainty_loss = data['upper_uncertainty_loss']
    below_uncertainty_loss = data['below_uncertainty_loss']
    p_values = data['p_values']

    all_go = [go.Scatter(x=x, y=loss, mode="lines", name='Loss',line=dict(color="#3399FF")),
              go.Scatter(x=x, y=threshold, mode="lines", name='Threshold',line=dict(color="yellow")),
              go.Scatter(x=x, y=upper_uncertainty_loss, mode="lines", name='Upper_uncertainty_loss',line=dict(color="#CCE5FF")),
              go.Scatter(x=x, y=below_uncertainty_loss, mode="lines", name='Below_uncertainty_loss',line=dict(color="#CCE5FF")),
              go.Scatter(x=x, y=p_values, mode="lines", name='p_values',line=dict(color="#FF66B2"))]
    return all_go
#############################################################################################################################
def main():
    # start agent network server
    agentNetwork = AgentNetwork()

    gen_agent_test1 = agentNetwork.add_agent(name="Sensor1", agentType=SineGeneratorAgent, log_mode=False)
    gen_agent_test2 = agentNetwork.add_agent(name="Sensor2",agentType=SineGeneratorAgent, log_mode=False)
    gen_agent_test3 = agentNetwork.add_agent(name="Sensor3",agentType=SineGeneratorAgent, log_mode=False)


    aggregator_agent = agentNetwork.add_agent(agentType=Aggregator)
    predictor_agent = agentNetwork.add_agent(agentType=Predictor)

    monitor_agent_1 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=100,log_mode=False)
    monitor_agent_2 =  agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=100,log_mode=False)

    gen_agent_test1.init_parameters(sensor_buffer_size=10)
    gen_agent_test1.init_agent_loop(loop_wait=.01)

    gen_agent_test2.init_parameters(sensor_buffer_size=10, scale_amplitude=.3)
    gen_agent_test2.init_agent_loop(loop_wait=.01)

    gen_agent_test3.init_parameters(sensor_buffer_size=10, scale_amplitude=.6)
    gen_agent_test3.init_agent_loop(loop_wait=.01)


    predictor_agent.init_parameters(100,"withLSTM") #define train_size aand machine learning model types(withLSTM or withoutLSTM)

    monitor_agent_1.init_parameters(plot_filter=['Time','y1','y2','y3'],custom_plot_function=custom_create_monitor_graph_actualdata)
    monitor_agent_2.init_parameters(plot_filter=['Time', 'loss', 'threshold', 'upper_uncertainty_loss','below_uncertainty_loss','p_values'],
                                    custom_plot_function=custom_create_monitor_graph_calculation)

    #bind agents
    agentNetwork.bind_agents(gen_agent_test1, aggregator_agent)
    agentNetwork.bind_agents(gen_agent_test2, aggregator_agent)
    agentNetwork.bind_agents(gen_agent_test3, aggregator_agent)

    agentNetwork.bind_agents(aggregator_agent, predictor_agent)
    agentNetwork.bind_agents(aggregator_agent, monitor_agent_1)
    agentNetwork.bind_agents(predictor_agent, monitor_agent_2)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()


# Done and finish