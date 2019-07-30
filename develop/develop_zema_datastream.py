from DataStreamMET4FOF import DataStreamMET4FOF
import numpy as np
import pandas as pd
import h5py


class ZEMA_DataStream(DataStreamMET4FOF):
    def __init__(self):
        f = h5py.File("F:/PhD Research/Github/agentMet4FoF/dataset/Sensor_data_2kHz.h5", 'r')

        # Order of sensors in the picture is different from the order in imported data, which will be followed.
        self.offset=[0, 0, 0, 0, 0.00488591, 0.00488591, 0.00488591,  0.00488591, 1.36e-2, 1.5e-2, 1.09e-2]
        self.gain=[5.36e-9, 5.36e-9, 5.36e-9, 5.36e-9, 3.29e-4, 3.29e-4, 3.29e-4, 3.29e-4, 8.76e-5, 8.68e-5, 8.65e-5]
        self.b=[1, 1, 1, 1, 1, 1, 1, 1, 5.299641744, 5.299641744, 5.299641744]
        self.k=[250, 1, 10, 10, 1.25, 1, 30, 0.5, 2, 2, 2]
        self.units=['[Pa]', '[g]', '[g]', '[g]', '[kN]', '[bar]', '[mm/s]', '[A]', '[A]', '[A]', '[A]']
        self.labels = ['Microphone', 'Vibration plain bearing','Vibration piston rod','Vibration ball bearing', 'Axial force','Pressure','Velocity','Active current','Motor current phase 1', 'Motor current phase 2','Motor current phase 3']
        #prepare sensor data
        list(f.keys())
        data= f['Sensor_Data']
        data= data[:,:,:data.shape[2]-1] #drop last cycle
        data_inputs_np = np.zeros([data.shape[2],data.shape[1],data.shape[0]])
        for i in range(data.shape[0]):
            sensor_dt = data[i].transpose()
            data_inputs_np[:,:,i] = sensor_dt

        #prepare target var
        target=list(np.zeros(data_inputs_np.shape[0]))          # Making the target list which takes into account number of cycles, which-
        for i in range(data_inputs_np.shape[0]):                # goes from 0 to 100, and has number of elements same as number of cycles.
            target[i]=(i/(data_inputs_np.shape[0]-1))*100

        target_matrix = pd.DataFrame(target)        # Transforming list "target" into data frame "target matrix"
        data_inputs_np = self.convert_SI(data_inputs_np)
        self.set_data_source(x=data_inputs_np, y=target_matrix)

    def convert_SI(self, sensor_ADC):
        sensor_SI = sensor_ADC
        for i in range(sensor_ADC.shape[2]):
            sensor_SI[:,:,i]=((sensor_ADC[:,:,i]*self.gain[i])+self.offset[i])*self.b[i]*self.k[i]
        return sensor_SI


"""
Examples
--------
if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType=ZEMA_DataStreamAgent)
    dummy_agent = agentNetwork.add_agent(agentType=AgentMET4FOF)

    #connect agents by either way:
    agentNetwork.bind_agents(gen_agent, dummy_agent)


    gen_agent.init_agent_loop(5)
    # # set all agents states to "Running"
    agentNetwork.set_running_state()
"""