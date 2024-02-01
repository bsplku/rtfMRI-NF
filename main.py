import json
import math
import os
import sys

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import *
from PySide6.QtWidgets import *

import experimental_parameters
from ui_main import Ui_MainWindow

class ExperimentMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_root = os.path.dirname(os.path.abspath(__file__)) # all path variables should be absolute paths.
        
        # placeholders for TaskMainWindows.
        self.nrtwm_task_window = None
        self.rt_task_window = None

        # load the ui_main.py
        # note that we need to run `pyside6-uic main.ui > ui_main.py`
        # after editing the main.ui file.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # TODO: Un-comment after finish
        self.ui.box_date.setDate(QDate.currentDate()) # set current date. 

        # set experimental parameters.
        self.params = {}
        self.check_and_register_experimental_paths()
        self.register_experimental_parameters()

        # review volumes in each RT/NRT task.
        self.review_volumes_for_tasks(task='rt')

        # register button handlers.
        self.ui.btn_nrt.clicked.connect(self.start_nrt_task)
        self.ui.btn_rt.clicked.connect(self.start_rt_task)
        self.ui.btn_sham_path.clicked.connect(self.set_sham_data_path)
        
    def check_and_register_experimental_paths(self):
        all_path = {}

        for key, key_path in experimental_parameters.PATH.items():
            abs_key_path = os.path.abspath(key_path)

            if not os.path.exists(abs_key_path):
                message = f'The path for {key} ({abs_key_path}) does not exist. Please check the experimental_parameters.py file.'
                response = QMessageBox.critical(self, 'Path Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)
            
            all_path[key] = abs_key_path
        
        self.params['PATH'] = all_path
    
    def register_experimental_parameters(self):
        self.params['MRI'] = experimental_parameters.MRI.copy()
        self.params['RT'] = experimental_parameters.RT.copy()
        self.params['NRT'] = experimental_parameters.NRT.copy()
    
    def review_volumes_for_tasks(self, task='rt'):
        def get_dict_or_error(dict_name, key):
            try:
                dict_to_lookup = self.params[dict_name]
            except KeyError:
                message = f'The parameter group ({dict_name}) was not registered. Please check the experimental_parameters.py file.'
                response = QMessageBox.critical(self, 'Parameter Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)
            
            try:
                return dict_to_lookup[key]
            except KeyError:
                message = f'The parameter ({key}) does not exist in the parameter group ({dict_name}). Please check the experimental_parameters.py file.'
                response = QMessageBox.critical(self, 'Parameter Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)
        
        TR = get_dict_or_error('MRI', 'tr')
        if task == 'rt':
            ins_duration = get_dict_or_error('RT', 'ins_duration') 
            fix_duration = get_dict_or_error('RT', 'fix_duration') 
            ready_duration = get_dict_or_error('RT', 'ready_duration')
            n_blocks_per_modality = get_dict_or_error('RT', 'n_blocks_per_modality')
            cue_duration_per_block = get_dict_or_error('RT', 'cue_duration_per_block')
            # change_cue_duration = get_dict_or_error('RT', 'change_cue_duration')
            fixation_duration_per_block = get_dict_or_error('RT', 'fixation_duration_per_block')
            n_trials_per_block = get_dict_or_error('RT', 'n_trials_per_block')
            trial_duration = get_dict_or_error('RT', 'trial_duration')
            trial_isi_duration = get_dict_or_error('RT', 'trial_isi_duration')
            feedback_duration_per_block = get_dict_or_error('RT', 'feedback_duration_per_block')
            inter_block_duration = get_dict_or_error('RT', 'inter_block_duration')

            rt_volumes = math.ceil((ins_duration + fix_duration + ready_duration + \
                4 * n_blocks_per_modality * (cue_duration_per_block + fixation_duration_per_block +  (n_trials_per_block - 1) * (trial_duration + trial_isi_duration) + (trial_duration + inter_block_duration))) / TR)
            self.params['RT']['total_volumes'] = rt_volumes # register RT total volumes in self.RT
            
            self.ui.label_rt_volumes.setText(f'{rt_volumes}')
        else:
            raise(ValueError(f'Unknown task ({task}).'))

    def prepare_tasks(self, task='rt'):
        subject = self.ui.box_subject.text()
        date = self.ui.box_date.text()

        if task == 'nrt':
            mri_series = int(self.ui.box_series_nrt.text())
            run_order = self.ui.run_nrt.currentText()
        elif task == 'rt':
            mri_series = int(self.ui.box_series_rt.text())
            run_order = self.ui.run_rt.currentText()
        else:
            raise(ValueError(f'Unknown task ({task}).'))

        subject_dir_name = f'{date}.{date}_{subject}.{date}_{subject}'
        series_dir_name = f'EPI_{mri_series:0>3}'
        subject_raw_path = os.path.join(self.params['PATH']['fmri_raw_path'], subject_dir_name)
        subject_series_path = os.path.join(self.params['PATH']['fmri_raw_path'], subject_dir_name, series_dir_name)
        subject_experiment_path = os.path.join(subject_raw_path, 'experiment')
        os.makedirs(subject_series_path, exist_ok=True)
        os.makedirs(subject_experiment_path, exist_ok=True)

        curr_parameter_dict = {'MRI': experimental_parameters.MRI, 'RT': experimental_parameters.RT, \
            'NRT': experimental_parameters.NRT, 'PATH': experimental_parameters.PATH}
        parameter_json_path = os.path.join(subject_experiment_path, 'experimental_parameters.json')
        
        # check experimental_parameters.json present and if not, save all parameters into a json file.
        if not os.path.exists(parameter_json_path):
            with open(parameter_json_path, 'w') as f:
                json.dump(curr_parameter_dict, f, indent=2)
        else:
            # if already present, 
            with open(parameter_json_path, 'r') as f:
                prev_parameter_dict = json.load(f)
            
            # load it and compare current parameters.
            # if two parameter sets are different, we can reset the file with the current parameter set,
            # or save the current parameter set with a new name.
            if prev_parameter_dict != curr_parameter_dict:
                response = QMessageBox.warning(self, 'Parameter Warning', \
                    'Current experimental parameters are different from the previous ones. Will you reset (replace) the previous setting file with the current parameters or save the current setting with a different name?', \
                        QMessageBox.Reset | QMessageBox.Save | QMessageBox.Cancel)
                
                if response == QMessageBox.Reset:
                    with open(parameter_json_path, 'w') as f:
                        json.dump(curr_parameter_dict, f, indent=2)

                elif response == QMessageBox.Save:
                    new_parameter_json_path = os.path.join(subject_experiment_path, \
                        f'experimental_parameters_{series_dir_name}_{task.upper()}_{run_order}.json')

                    with open(new_parameter_json_path, 'w') as f:
                        json.dump(curr_parameter_dict, f, indent=2)
                
                elif response == QMessageBox.Cancel:
                    sys.exit(0)
        
        # TODO: it seems volume_log_path is not used.
        if task == 'rt':
            # check etime, volume files present.
            etime_path = os.path.join(subject_experiment_path, \
                f'etime_{series_dir_name}_{task.upper()}_{run_order}.etime')
            volume_log_path = os.path.join(subject_experiment_path, \
                f'volume_{series_dir_name}_{task.upper()}_{run_order}.log')
            
            # if the etime or log file are already present,
            # we can replace it (continue) or exit the program for review (cancel). 
            if os.path.exists(etime_path) or os.path.exists(volume_log_path):
                response = QMessageBox.warning(self, 'Record Warning', \
                    'Data file(s) for this EPI series and run order are already present. Continue?', \
                        QMessageBox.Yes | QMessageBox.Cancel)
                
                if response == QMessageBox.Cancel:
                    sys.exit(0)
        
        elif task == 'nrt':
            etime_path = os.path.join(subject_experiment_path, \
                f'etime_{series_dir_name}_{task.upper()}_{run_order}.etime')
            
            video_history_path = os.path.join(subject_experiment_path, \
                f'video_history_{task.upper()}.txt')
            
            if os.path.exists(etime_path):
                response = QMessageBox.warning(self, 'Record Warning', \
                    'Data file(s) for this EPI series and run order are already present. Continue?', \
                        QMessageBox.Yes | QMessageBox.Cancel)
                
                if response == QMessageBox.Cancel:
                    sys.exit(0)

        # register run parameters.
        self.params['RUN'] = {}
        self.params['RUN']['subject'] = subject
        self.params['RUN']['date'] = date
        self.params['RUN']['mri_series'] = mri_series
        self.params['RUN']['run_order'] = run_order
        self.params['RUN']['subject_raw_path'] = subject_raw_path
        self.params['RUN']['subject_series_path'] = subject_series_path

        if task == 'rt':
            self.params['RUN']['etime_path'] = etime_path
            self.params['RUN']['volume_log_path'] = volume_log_path
            if 'SHAM' not in self.params:
                self.params['SHAM'] = None\
        
        elif task == 'nrt':
            self.params['RUN']['etime_path'] = etime_path
            self.params['RUN']['video_history_path'] = video_history_path
    
    def set_sham_data_path(self):
        response = QFileDialog.getOpenFileName(self, 'Open Data File', self.params['PATH']['fmri_raw_path'])
        data_filepath = response[0]

        if len(data_filepath) > 0: # file selected.
            try:
                with open(data_filepath, 'r') as f:
                    raw_etime_lines = [line.strip() for line in f.readlines()]
                
                only_task_lines = filter(lambda line: not '*' in line, raw_etime_lines)
                message_lines = list(map(lambda line: line.split('\t')[1], only_task_lines))

                sham_block_target_list = []
                sham_block_trial_list = []
                sham_block_feedback_list = []

                for message in message_lines:
                    if 'cue,target' in message:
                        sham_block_target_list.append(message.split('target:')[-1])

                        # check trial counts of the previous block
                        if len(sham_block_trial_list) > 0:
                            if not len(sham_block_trial_list[-1]) == self.params['RT']['n_trials_per_block']:
                                raise ValueError(f'Trial number mismatches at block {len(sham_block_trial_list)}.')
                        
                        sham_block_trial_list.append([])
                    
                    if ('image' in message) and ('sound' in message) and ('text' in message) and ('feedback' not in message):
                        _, image_id_raw, sound_id_raw, text_id_raw, _ = message.split(',')
                        image_id = int(image_id_raw.split(':')[-1])
                        sound_id = int(sound_id_raw.split(':')[-1])
                        text_id = int(text_id_raw.split(':')[-1])
                        sham_block_trial_list[-1].append((image_id, sound_id, text_id))
                    
                    if ('image' in message) and ('sound' in message) and ('text' in message) and ('feedback' in message):
                        _, image_prob_raw, sound_prob_raw, text_prob_raw = message.split(',')
                        image_prob = float(image_prob_raw.split(':')[-1])
                        sound_prob = float(sound_prob_raw.split(':')[-1])
                        text_prob = float(text_prob_raw.split(':')[-1])
                        sham_block_feedback_list.append((image_prob, sound_prob, text_prob))
                
                if len(sham_block_trial_list) > 0:
                    if not len(sham_block_trial_list[-1]) == self.params['RT']['n_trials_per_block']:
                        raise ValueError(f'Trial number mismatches at block {len(sham_block_trial_list)}.')
                
                image_block_number = len([1 for block in sham_block_target_list if block == 'image'])
                sound_block_number = len([1 for block in sham_block_target_list if block == 'sound'])
                text_block_number = len([1 for block in sham_block_target_list if block == 'text'])

                if image_block_number != self.params['RT']['n_blocks_per_modality']:
                    raise ValueError(f'Image block count mismatches.')
                
                if sound_block_number != self.params['RT']['n_blocks_per_modality']:
                    raise ValueError(f'Sound block count mismatches.')
                
                if text_block_number != self.params['RT']['n_blocks_per_modality']:
                    raise ValueError(f'Text block count mismatches.')

            except OSError:
                message = 'The selected etime file may be broken. Please check the file.'
                response = QMessageBox.critical(self, 'Data File Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)
            
            except ValueError as e:
                message = f'{e} Please check the file.'
                response = QMessageBox.critical(self, 'Data File Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)
            
            except Exception as e:
                print(e)
                message = 'An error occurred. Please check the terminal.'
                response = QMessageBox.critical(self, 'Data File Error!', message)
                if response == QMessageBox.Ok:
                    sys.exit(1)

            self.ui.label_sham_path.setText(f'... {data_filepath[-50:]}')
            self.params['SHAM'] = {}
            self.params['SHAM']['block_target_list'] = sham_block_target_list
            self.params['SHAM']['block_trial_list'] = sum(sham_block_trial_list, []) # flatten
            self.params['SHAM']['block_feedback_list'] = sham_block_feedback_list
        else:
            self.params['SHAM'] = None
            self.ui.label_sham_path.setText('None')

    # def start_nrt_task(self):
    #     self.prepare_tasks(task='nrt')
    #     self.ui.btn_nrt.setDisabled(True)

    #     from nrt.pages import NRTTaskMainWindow
    #     self.nrt_task_window = NRTTaskMainWindow(self, task_params=self.params.copy())            
    #     self.nrt_task_window.show()

    def start_rt_task(self):
        self.prepare_tasks(task='rt')

        from rt.pages import RTTaskMainWindow
        self.rt_task_window = RTTaskMainWindow(self, task_params=self.params.copy())
        self.rt_task_window.show()     

    def closeEvent(self, e):
        self.hide()
        e.accept()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    experiment_main_window = ExperimentMainWindow()
    experiment_main_window.show()
    sys.exit(app.exec())