import os
import re
import time
import threading
from queue import Queue, Empty
from multiprocessing import Process

import numpy as np
import torch
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from nibabel import load as nib_load
from scipy.io import loadmat
from scipy.special import softmax

from .preprocessing import Preprocessing

class AFNIPreprocess(QObject):
    registration_finished = Signal()
    baseline_finished = Signal()
    preprocessing_finished = Signal(int)

    def __init__(self, params, queue: Queue):
        super().__init__()
        self.shutdown_flag = threading.Event()
        self.afni_command = None
        self.params = params
        self.mri_series = self.params['RUN']['mri_series']
        self.subject_series_path = self.params['RUN']['subject_series_path']
        self.queue: Queue = queue
        self.task_process = None
    
    def registration_process_task(self):
        os.chdir(self.subject_series_path)
        self.afni_command = Preprocessing(self.params)
                
        self.afni_command.copy(in_file='/home/bspl/abin/MNI_EPI+tlrc', \
            out_file='MNI_EPI.nii')
        self.afni_command.run()

        self.afni_command.to3d()
        self.afni_command.run()
        try :
            self.afni_command.align_epi_anat()
            self.afni_command.run()
        finally:
            self.afni_command.copy(in_file=f'epi_{self.mri_series:0>3}_base_al+tlrc', \
                out_file=f'epi_{self.mri_series:0>3}_base_al.nii')
            self.afni_command.run()

    def baseline_process_task(self, volume_index, baseline_volume_index):
        os.chdir(self.subject_series_path)

        time.sleep(0.5)
        self.afni_command = Preprocessing(self.params, volume_number=volume_index)
        self.afni_command.to3d()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.alignment()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.volreg()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.blur()
        self.afni_command.run()
        time.sleep(0.1)

        if volume_index == baseline_volume_index:
            self.afni_command.tcat(baseline_volume=baseline_volume_index)
            self.afni_command.run()
            time.sleep(0.1)

        elif volume_index == baseline_volume_index + 1:
            self.afni_command.tcat(baseline_volume=baseline_volume_index)
            self.afni_command.run()
            time.sleep(0.1)

            self.afni_command.mean(baseline_volume=baseline_volume_index)
            self.afni_command.run()
            time.sleep(0.1)

            # self.afni_command.automask(baseline_volume=baseline_volume_index)
            # self.afni_command.run()
            # time.sleep(0.1)
    
    # def preprocess_process_task(self, volume_index):
    #     os.chdir(self.subject_series_path)

    #     time.sleep(0.5)
    #     self.afni_command = Preprocessing(self.params, volume_number=volume_index)
    #     self.afni_command.to3d()
    #     self.afni_command.run()
    #     time.sleep(0.1)

    #     self.afni_command.alignment()
    #     self.afni_command.run()
    #     time.sleep(0.1)

    #     self.afni_command.volreg()
    #     self.afni_command.run()
    #     time.sleep(0.1)

    #     self.afni_command.blur()
    #     self.afni_command.run()
    #     time.sleep(0.1)

    #     try:
    #         if volume_index <= 30 :                       
    #             self.afni_command.tcat()
    #             self.afni_command.run()
    #             time.sleep(0.1)

    #         if volume_index > 30:
    #             self.afni_command.scale()
    #             self.afni_command.run()
    #             time.sleep(0.1)
    #     except:
    #         self.afni_command.mean()
    #         self.afni_command.run()
    #         time.sleep(0.1)

    #         self.afni_command.automask()
    #         self.afni_command.run()
    #         time.sleep(0.1)
    
    def preprocess_process_task(self, volume_index, baseline_volume_index):
        os.chdir(self.subject_series_path)

        time.sleep(0.5)
        self.afni_command = Preprocessing(self.params, volume_number=volume_index)
        self.afni_command.to3d()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.alignment()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.volreg()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.blur()
        self.afni_command.run()
        time.sleep(0.1)

        self.afni_command.scale(baseline_volume=baseline_volume_index)
        self.afni_command.run()
        time.sleep(0.1)


    # def run_preprocessing(self):
    #     while not self.shutdown_flag.is_set():
    #         try:
    #             time.sleep(0.01)
    #             volume_index = self.queue.get(timeout=0.01) 
    #         except Empty:
    #             continue

    #         if volume_index == 'registration':
    #             self.task_process = Process(target=self.registration_process_task, daemon=True)
    #             self.task_process.start()
    #             self.task_process.join()
    #             # self.registration_process_task()
                
    #             time.sleep(0.01)
    #             self.registration_finished.emit()
                    
    #         else:
    #             self.task_process = Process(target=self.preprocess_process_task, args=(volume_index,), daemon=True)
    #             self.task_process.start()
    #             self.task_process.join() 
    #             # self.preprocess_process_task(volume_index)               

    #             time.sleep(0.01)
    #             self.preprocessing_finished.emit(volume_index)
    
    def run_preprocessing(self):
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(0.01)
                command, current_volume_index, baseline_volume_index = self.queue.get(timeout=0.01) 
            except Empty:
                continue

            if command == 'registration':
                self.task_process = Process(target=self.registration_process_task, daemon=True)
                self.task_process.start()
                self.task_process.join()
                # self.registration_process_task()
                
                time.sleep(0.01)
                self.registration_finished.emit()
            
            
            elif command == 'baseline':
                self.task_process = Process(target=self.baseline_process_task, args=(current_volume_index, baseline_volume_index), daemon=True)
                self.task_process.start()
                self.task_process.join()

                time.sleep(0.01)
                if current_volume_index == baseline_volume_index + 1:
                    self.baseline_finished.emit()
                    
            else:
                self.task_process = Process(target=self.preprocess_process_task, args=(current_volume_index, baseline_volume_index), daemon=True)
                self.task_process.start()
                self.task_process.join() 
                # self.preprocess_process_task(volume_index)               

                time.sleep(0.01)
                self.preprocessing_finished.emit(current_volume_index)
    
    def cleanup(self):
        if self.task_process is not None:
            self.task_process.join()
            self.task_process.terminate()
            self.task_process.join()
        self.shutdown_flag.set()

class AFNIPreprocessor:
    def __init__(self, params, registration_handler, baseline_handler, preprocessing_handler):
        self.preprocess_queue = Queue()
        self._afni_preprocess = AFNIPreprocess(params, queue=self.preprocess_queue)
        self._thread = QThread()
        self._afni_preprocess.registration_finished.connect(registration_handler)
        self._afni_preprocess.baseline_finished.connect(baseline_handler)
        self._afni_preprocess.preprocessing_finished.connect(preprocessing_handler)
        self._afni_preprocess.moveToThread(self._thread)
        self._thread.started.connect(self._afni_preprocess.run_preprocessing)
        self._thread.start()
    
    # def queue_preprocessing_task(self, volume_index=None, registration=False):
    #     if registration and volume_index is None:
    #         self.preprocess_queue.put('registration')
    #     elif not registration and volume_index is not None:
    #         self.preprocess_queue.put(volume_index)
    #     else:
    #         raise ValueError('only volume_index or registration parameter should be set.')
    
    def queue_preprocessing_task(self, command, current_volume_index=None, baseline_volume_index=None):
        # if registration and volume_index is None:
        #     self.preprocess_queue.put('registration')
        # elif not registration and volume_index is not None:
        #     self.preprocess_queue.put(volume_index)
        # else:
        #     raise ValueError('only volume_index or registration parameter should be set.')
        self.preprocess_queue.put((command, current_volume_index, baseline_volume_index))
    
    def quit_preprocessor(self):
        if self._afni_preprocess is not None and self._thread is not None:
            self._afni_preprocess.cleanup()
            self._afni_preprocess.deleteLater()
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()

            self._afni_preprocess = None
            self._thread = None

#################################################################################
#################################################################################

# defined torch DNN model class by hailey
class DNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, params):

        super(DNN, self).__init__()
        self.n_hid_nodes = params['n_hid_nodes']
        self.n_hid_layers = len(self.n_hid_nodes)
        self.dropout_rate = params['dropout_rate']

        self.nn = [None]*(self.n_hid_layers+1)
        self.nn_bn = [None]*self.n_hid_layers
        self.dropout = [None]*self.n_hid_layers

        self.nn[0] = torch.nn.Linear(input_dim, self.n_hid_nodes[0])
        self.nn_bn[0] = torch.nn.BatchNorm1d(self.n_hid_nodes[0])
        self.dropout[0] = torch.nn.Dropout(p=self.dropout_rate[0])
        for i_hid in range(1,self.n_hid_layers):
            self.nn[i_hid] = torch.nn.Linear(self.n_hid_nodes[i_hid-1], self.n_hid_nodes[i_hid])
            self.nn_bn[i_hid] = torch.nn.BatchNorm1d(self.n_hid_nodes[i_hid])
            self.dropout[i_hid] = torch.nn.Dropout(p=self.dropout_rate[i_hid-1])
        self.nn[-1] = torch.nn.Linear(self.n_hid_nodes[-1], output_dim)

        self.nn = torch.nn.ModuleList(self.nn)
        self.nn_bn = torch.nn.ModuleList(self.nn_bn)
        self.dropout = torch.nn.ModuleList(self.dropout)

        # self.sigmoid = nn.Sigmoid()
        self.act_func = torch.nn.ReLU() #nn.Tanh()
        self.weights_init()

    def forward(self, x):

        for i_hid in range(self.n_hid_layers):
            x = self.nn[i_hid](x)
            x = self.nn_bn[i_hid](x)
            x = self.act_func(x)
            x = self.dropout[i_hid](x)
        out = self.nn[-1](x)
        # x = self.nn[-1](x)
        # out = self.sigmoid(x)

        return out

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                torch.nn.init.normal_(m.bias, std=0.01)

class TorchDNNModel(QObject):
    probability_response = Signal(int, float, float, float)

    def __init__(self, params, queue: Queue):
        super().__init__()
        self.torch_dnn_model_path = params['PATH']['torch_dnn_model_path']
        self.mask_path = params['PATH']['mask_path']
        self.mri_series = params['RUN']['mri_series']
        self.subject_series_path = params['RUN']['subject_series_path']
        self.queue = queue
        self.shutdown_flag = threading.Event()
        self.load_mask_image()
        self.load_dnn_models()
    
    def load_mask_image(self):
        mask_image = nib_load(self.mask_path)
        self.mask = np.where(mask_image.get_fdata())
        del mask_image
    
    def load_dnn_models(self):
        torch_state_dict = torch.load(self.torch_dnn_model_path, map_location='cpu')
        model_params = {'n_hid_nodes': [512, 512], 'dropout_rate': [0.4, 0.4]}
        self.torch_model = DNN(input_dim=52470, output_dim=3, params=model_params).to('cpu')
        self.torch_model.load_state_dict(torch_state_dict)
    
    def load_fmri_image(self, volume_index=None):
        preprocessed_file_path = os.path.join(self.subject_series_path, \
            f'epi_{self.mri_series:0>3}_{volume_index:0>3}_scale.nii')
        new_image = nib_load(preprocessed_file_path)
        new_image_array = new_image.get_fdata()
        masked_image = new_image_array[self.mask]
        self.image = np.nan_to_num(masked_image, nan=0.0)
        self.volume = volume_index
    
    def predict_probability(self):
        with torch.no_grad():
            self.torch_model.eval()
            torch_input = torch.tensor(self.image.reshape(1, -1), device='cpu', dtype=torch.float32)
            torch_output = self.torch_model(torch_input)

            image_prob, sound_prob, text_prob = softmax(torch_output)[0]
            self.probability_response.emit(self.volume, image_prob, sound_prob, text_prob)
    
    def run_inference(self):
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(0.01)
                preprocessed_volume = self.queue.get(timeout=0.01) 
            except Empty:
                continue
            
            try:
                self.load_fmri_image(preprocessed_volume)
                time.sleep(0.01)
                self.predict_probability()
                time.sleep(0.1)
            except:
                self.probability_response.emit(preprocessed_volume, 1/3, 1/3, 1/3)
    
    def cleanup(self):
        self.shutdown_flag.set()
        del self.mask
        del self.torch_model


class TorchDNNPredictor:
    def __init__(self, parent_params, prediction_handler):
        self.prediction_queue = Queue()
        self._dnn = TorchDNNModel(params=parent_params, queue=self.prediction_queue)
        self._thread = QThread()
        self._dnn.probability_response.connect(prediction_handler)
        self._dnn.moveToThread(self._thread)
        self._thread.started.connect(self._dnn.run_inference)
        self._thread.start()
    
    def run_prediction(self, preprocessed_volume):
        # if self._dnn is not None:
        #     try:
        #         self._dnn.load_fmri_image(preprocessed_volume)
        #         self._dnn.predict_probability()
        #     except:
        #         self._dnn.probability_response.emit(preprocessed_volume, 1/3, 1/3, 1/3)
        # time.sleep(0.1) # sleep processing a short time to prevent block

        if preprocessed_volume is not None:
            self.prediction_queue.put(preprocessed_volume)
        else:
            raise ValueError('preprocessed volume should be set.')
    
    def clear_queue(self):
        while not self.prediction_queue.empty():
            try:
                self.prediction_queue.get(block=False)
            except Empty:
                continue

            self.prediction_queue.task_done()

    
    def quit_dnn_predictor(self):
        if self._dnn is not None and self._thread is not None:
            self._dnn.cleanup()
            self._dnn.deleteLater()
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()

            self._dnn = None
            self._thread = None


# defined torch CNN model class by hailey
class CNN(torch.nn.Module):
    # https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
    def __init__(self, input_dim, output_dim, params):

        super(CNN, self).__init__()

        self.act_func = torch.nn.LeakyReLU()
        self.n_channs = params['n_channs']
        self.conv_kernels = params['conv_kernels']
        self.conv_strides = params['conv_strides']
        self.maxpool_kernels = params['maxpool_kernels']
        # self.maxpool_strides = params['maxpool_strides']
        self.n_conv_layers = len(self.conv_kernels)

        self.n_hid_nodes = params['n_hid_nodes']
        self.n_hid_layers = len(self.n_hid_nodes)
        self.dropout_rate = params['dropout_rate']

        self.hid_dim = [None]*self.n_conv_layers
        self.conv_nn = [None]*self.n_conv_layers
        self.nn = [None]*(self.n_hid_layers+1)
        self.nn_bn = [None]*self.n_hid_layers
        self.dropout = [None]*self.n_hid_layers

        # e.g., 16x16x16 -> 13x13x13 -> 7x7x7
        self.conv_nn[0] = self._conv_layer_set(1, self.n_channs[0], conv_kernel=self.conv_kernels[0], conv_stride=self.conv_strides[0], maxpool_kernel=self.maxpool_kernels[0], maxpool_stride=self.maxpool_kernels[0])
        self.hid_dim[0] = np.ceil((np.array(input_dim)-self.conv_kernels[0])/self.maxpool_kernels[0]).astype(int)
        for i_conv in range(1,self.n_conv_layers):
            # 7x7x7 -> 4x4x4 -> 2x2x2
            self.conv_nn[i_conv] = self._conv_layer_set(self.n_channs[i_conv-1], self.n_channs[i_conv], conv_kernel=self.conv_kernels[i_conv], conv_stride=self.conv_strides[i_conv], maxpool_kernel=self.maxpool_kernels[i_conv], maxpool_stride=self.maxpool_kernels[i_conv])
            self.hid_dim[i_conv] = np.ceil((self.hid_dim[i_conv-1]-self.conv_kernels[i_conv])/self.maxpool_kernels[i_conv]).astype(int)

        print(self.hid_dim)

        self.nn[0] = torch.nn.Linear(np.prod(self.hid_dim[-1])*self.n_channs[-1], self.n_hid_nodes[0])
        self.nn_bn[0] = torch.nn.BatchNorm1d(self.n_hid_nodes[0])
        self.dropout[0] = torch.nn.Dropout(p=self.dropout_rate[0])
        for i_hid in range(1,self.n_hid_layers):
            self.nn[i_hid] = torch.nn.Linear(self.n_hid_nodes[i_hid-1], self.n_hid_nodes[i_hid])
            self.nn_bn[i_hid] = torch.nn.BatchNorm1d(self.n_hid_nodes[i_hid-1])
            self.dropout[i_hid] = torch.nn.Dropout(p=self.dropout_rate[i_hid-1])
        self.nn[-1] = torch.nn.Linear(self.n_hid_nodes[-1], output_dim)

        self.conv_nn = torch.nn.ModuleList(self.conv_nn)
        self.nn = torch.nn.ModuleList(self.nn)
        self.nn_bn = torch.nn.ModuleList(self.nn_bn)
        self.dropout = torch.nn.ModuleList(self.dropout)

        self.weights_init()


    def _conv_layer_set(self, in_c, out_c, conv_kernel=3, conv_stride=1, maxpool_kernel=2, maxpool_stride=2):
        conv_layer = torch.nn.Sequential(
                                    torch.nn.Conv3d(in_c, out_c, kernel_size=(conv_kernel, conv_kernel, conv_kernel), stride=conv_stride, padding=0),
                                    self.act_func,
                                    torch.nn.MaxPool3d(kernel_size=(maxpool_kernel, maxpool_kernel, maxpool_kernel), stride=maxpool_stride),
                                )
        return conv_layer

    def forward(self, x):

        # torch.Size([#, 1, 61, 73, 61])
        x = self.conv_nn[0](x)
        # torch.Size([#, 32, 29, 35, 29])
        for i_conv in range(1,self.n_conv_layers):
            x = self.conv_nn[i_conv](x)
            # torch.Size([#, 64, 13, 16, 13])
        x = x.view(x.size(0), -1)
        # torch.Size([#, 173056])

        for i_hid in range(self.n_hid_layers):
            x = self.nn[i_hid](x)
            x = self.nn_bn[i_hid](x)
            x = self.act_func(x)
            x = self.dropout[i_hid](x)
        out = self.nn[-1](x)
        return out

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                torch.nn.init.normal_(m.bias, std=0.01)

            # https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
            elif isinstance(m, torch.nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

            # elif isinstance(m, nn.LayerNorm):
            #     m.bias.data.zero_()
            #     m.weight.data.fill_(1.0)

class TorchCNNModel(QObject):
    probability_response = Signal(int, float, float, float)

    def __init__(self, params, queue: Queue):
        super().__init__()
        self.torch_cnn_model_path = params['PATH']['torch_cnn_model_path']
        self.mask_path = params['PATH']['mask_path']
        self.mri_series = params['RUN']['mri_series']
        self.subject_series_path = params['RUN']['subject_series_path']
        self.queue = queue
        self.shutdown_flag = threading.Event()
        self.load_mask_image()
        self.load_cnn_models()
    
    def load_mask_image(self):
        mask_image = nib_load(self.mask_path)
        self.mask = mask_image.get_fdata().astype(bool)
        del mask_image
    
    def load_cnn_models(self):
        torch_state_dict = torch.load(self.torch_cnn_model_path, map_location='cpu')
        model_params = {'n_channs': [16, 16], 'conv_kernels': [3, 3], 'conv_strides': [1, 1], 'maxpool_kernels': [2, 2], 
                        'n_hid_nodes': [128], 'dropout_rate': [0.3]}
        self.torch_model = CNN(input_dim=(61, 73, 61), output_dim=3, params=model_params).to('cpu')
        self.torch_model.load_state_dict(torch_state_dict)
    
    def load_fmri_image(self, volume_index=None):
        preprocessed_file_path = os.path.join(self.subject_series_path, \
            f'epi_{self.mri_series:0>3}_{volume_index:0>3}_scale.nii')
        new_image = nib_load(preprocessed_file_path)
        new_image_array = new_image.get_fdata()
        masked_image = new_image_array * self.mask # new_image_array[self.mask]
        self.image = np.nan_to_num(masked_image, nan=0.0)
        self.volume = volume_index
    
    def predict_probability(self):
        with torch.no_grad():
            self.torch_model.eval()
            torch_input = torch.tensor(np.expand_dims(self.image, axis=(0, 1)), device='cpu', dtype=torch.float32)
            print(torch_input.size())
            torch_output = self.torch_model(torch_input)
            print(torch_output)

            image_prob, sound_prob, text_prob = softmax(torch_output)[0]
            self.probability_response.emit(self.volume, image_prob, sound_prob, text_prob)
    
    def run_inference(self):
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(0.01)
                preprocessed_volume = self.queue.get(timeout=0.01) 
            except Empty:
                continue
            
            try:
                self.load_fmri_image(preprocessed_volume)
                time.sleep(0.01)
                self.predict_probability()
                time.sleep(0.1)
            except Exception as e:
                self.probability_response.emit(preprocessed_volume, 1/3, 1/3, 1/3)
    
    def cleanup(self):
        self.shutdown_flag.set()
        del self.mask
        del self.torch_model


class TorchCNNPredictor:
    def __init__(self, parent_params, prediction_handler):
        self.prediction_queue = Queue()
        self._cnn = TorchCNNModel(params=parent_params, queue=self.prediction_queue)
        self._thread = QThread()
        self._cnn.probability_response.connect(prediction_handler)
        self._cnn.moveToThread(self._thread)
        self._thread.started.connect(self._cnn.run_inference)
        self._thread.start()
    
    def run_prediction(self, preprocessed_volume):
        # if self._dnn is not None:
        #     try:
        #         self._dnn.load_fmri_image(preprocessed_volume)
        #         self._dnn.predict_probability()
        #     except:
        #         self._dnn.probability_response.emit(preprocessed_volume, 1/3, 1/3, 1/3)
        # time.sleep(0.1) # sleep processing a short time to prevent block

        if preprocessed_volume is not None:
            self.prediction_queue.put(preprocessed_volume)
        else:
            raise ValueError('preprocessed volume should be set.')
    
    def clear_queue(self):
        while not self.prediction_queue.empty():
            try:
                self.prediction_queue.get(block=False)
            except Empty:
                continue

            self.prediction_queue.task_done()

    
    def quit_dnn_predictor(self):
        if self._cnn is not None and self._thread is not None:
            self._cnn.cleanup()
            self._cnn.deleteLater()
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()

            self._cnn = None
            self._thread = None