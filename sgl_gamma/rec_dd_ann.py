import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from astropy.io import ascii
from astropy.table import unique 
from tensorflow import keras
import os 
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from astropy.constants import c
import numpy as np
from tensorflow.keras import layers
from astropy.table import Table

import sys

c = c.to('km/s')

class ANN:

    def __init__(self, 
                 lens_table_path,
                 path_project,
                 data=None, 
                 model=None, metric='mse',
                 epochs= 300,  
                 save_model_as='weights_ANN.keras',
                 output_folder =None
                   ):
        
        self.path_project = path_project
        
        self.lens_table_path = lens_table_path
        if data is None:
            self.data = self.read_data()
        self.model = model
        self.metric = metric
        self.loss_list = ['mae', 'mse']
        self.epochs = epochs
        self.results = self.data[1350:] 
        if output_folder is None:
            self.output_folder = os.path.join(self.path_project, 'Output', 'ANN') 
        else:
            self.output_folder = os.path.join(self.path_project, 'Output', output_folder) 
        self.save_model_as = save_model_as
        self.path_output_table = os.path.join(self.output_folder, 'SGLTable_ANN.csv')

        if not os.path.exists(self.output_folder):
            print(f'making dir {self.output_folder}')
            os.makedirs(self.output_folder, exist_ok=True)

    #def luminosity_distance():
    def read_data(self):
        #print(os.path.join(self.path_project, 'Data' , 'Pantheon+SH0ES.dat'))
        data = ascii.read(os.path.join(self.path_project, 'Data' , 'Pantheon+SH0ES.dat'))
        data = unique(data,keys='CID')
        data.sort('mBERR')            
        #print(data)
        return data

    def distance_ratio(self, zl, zs):
    # Use the model to predict dl and ds
        dl = self.model.predict(zl)[:,0]
        dl_e = self.model.predict(zl)[:,1]
        
        ds = self.model.predict(zs)[:,0]
        ds_e = self.model.predict(zs)[:,1]
        
        dd = 1- (dl/ds)*((1+zs)/(1+zl))
        
        #dd_e = (1+zs)/(1+zl)*(1/ds)*np.sqrt(dl_e**2+(ds_e/ds)**2)
        dd_e = (1+zs)/(1+zl)*(1/ds)*np.sqrt(dl_e**2+dl**2.*(ds_e/ds)**2)

        return dd, dd_e


    def train_test(self,data, y):

        X = data['zHD'][:1350]
        Y = y[:1350]
        X_val = data['zHD'][1200:1350]
        Y_val = y[1200:1350]
        X_test = data['zHD'][1350:]
        Y_test = y[1350:]
        return X,Y, X_val, Y_val, X_test, Y_test

    def model_ANN(self,loss):

        input_shape = 1
        inputs=Input(shape=(input_shape,),name='input_layer')

        x =  Dense(2048, activation = 'elu',kernel_initializer="glorot_uniform",name='dense1')(inputs)
        # x = layers.Dropout(0.5)(x)

        output = layers.Dense(2)(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,decay_steps=1000,decay_rate=0.9)
        opt1=tf.keras.optimizers.Adam(learning_rate=lr_schedule1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(optimizer=opt1, loss=loss, metrics=[self.metric])
        # model.compile(optimizer=opt1, loss=tf.keras.losses.Huber(delta=8000.0), metrics=['mse'])
        #model.summary()
        return model

    def train_model(self,X,Y,X_val, Y_val):
        
        for loss in self.loss_list:
            self.model = self.model_ANN(loss)
            history = self.model.fit(X,Y,epochs=self.epochs , 
                        validation_data=(X_val,Y_val),batch_size=128, verbose=1)  
          
        self.plot_loss(history)
        self.model.save(os.path.join(self.output_folder, self.save_model_as), overwrite=True)

    def test_model(self, X_test):   

        Y_pred = self.model.predict(X_test)
        self.results['pred_luminosity_distance'] = Y_pred[:,0]
        self.results['pred_luminosity_distance_error'] = Y_pred[:,1]
        self.results.sort('zHD')

    def plot_loss(self,history):

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        plt.savefig(os.path.join(self.output_folder, 'Plot_loss.png'))
        plt.show(block=False)

        
    def rec_curve(self, rec_point_number):
        z_list = np.linspace(0,3.,rec_point_number+1)[1:]
        dl = self.model.predict(z_list)[:,0]
        dl_e = self.model.predict(z_list)[:,1]
        da = dl/(1.+z_list)**2.0
        da_e = np.sqrt(1./(1.+z_list)**4.0*dl_e**2.0)
        rec_curve = np.c_[z_list,da,da_e]
        rec_curve_output = os.path.join(self.output_folder, 'rec_curve.npy')
        np.save(rec_curve_output,rec_curve)
        print("Diameter dustance saved!")
        
    def main(self,):

        
        luminosity_distance = 10**((self.data['MU_SH0ES']-25)/5)
        luminosity_distance_error = ((10**((self.data['MU_SH0ES']-25)/5)*np.log(10))*self.data['MU_SH0ES_ERR_DIAG'])/5
        distance = np.vstack([luminosity_distance,luminosity_distance_error]).T
        X,Y, X_val, Y_val, X_test, Y_test = self.train_test(self.data, distance)

        if os.path.isfile(os.path.join(self.output_folder, self.save_model_as)):
            print('Found model weights, loading the weights')
            self.model = keras.models.load_model(os.path.join(self.output_folder, self.save_model_as), compile=False, custom_objects={'mse': 'mse'} )
            #self.model.compile(loss=self.loss_list[1])  

        else:
            sys.exit()
            print('Training the model')
            self.train_model(X,Y,X_val, Y_val)
        
        format = self.lens_table_path.split('.')[-1]

        SL = Table.read(self.lens_table_path, format=format)
        dd, dd_e = self.distance_ratio(SL['zl'],SL['zs'])
        SL['dd_ANN'] = dd
        SL['dd_error_ANN'] = dd_e
        print(f"Creating table with distance ratio from ANN predictions in { self.path_output_table}")
        SL.write(self.path_output_table, overwrite = True, format ='csv')