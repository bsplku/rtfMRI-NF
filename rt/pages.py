import os
import random
import textwrap
import sys
from datetime import datetime

import numpy as np
import matplotlib.patheffects as path_effects
import pygame
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PIL import Image
from pytz import timezone

from common.base import TaskMainWindow, TaskPage
from common.pages import SyncWaitPage, ReadyPage, FixationPage, EndPage
from common.resources import Waiter

from .resources import AFNIPreprocessor, TorchCNNPredictor
from .utils import random_shuffle_modalities, random_sample_coco_id

class RTConstants(object):
    # constants for sync status.
    SYNC_ZERO = 'zero'
    SYNC_REGISTER = 'register'
    SYNC_BASELINE = 'baseline_create'
    SYNC_RT = 'rt'

    # constants for block status.
    BLOCK_CUE = 'block_cue'
    BLOCK_FIXATION = 'block_fixation'
    BLOCK_TRIAL = 'block_trial'
    BLOCK_ISI = 'block_isi'
    BLOCK_IBI = 'block_ibi' # inter-block interval

class RTTaskMainWindow(TaskMainWindow):
    def set_task_parameters(self):
        super().set_task_parameters()

        # sync status.
        self._sync_status = RTConstants.SYNC_ZERO

        # DNN prediction queue
        self._image_probability_queue = []

        # registration flag
        self._registration_started = False
        self._registration_finished = False

        # baseline flag
        self._baseline_volume = None
        self._new_baseline_accept = True

        # preprocessing flag
        self._preprocessing_on = False

        # DNN prediction flag
        self._dnn_prediction_on = False

        # etime message set (to check duplication; the `in` operator takes O(1).)
        self._etime_message_set = set()

        # init pyagme
        pygame.init()

    def init_ui(self):
        self.pages = [SyncWaitPage(self), RTInstructionPage(self), RTFixationPage(self), RTReadyPage(self), \
                      RTCOCOTaskPage(self), RTEndPage(self)]
    
    def load_resource(self):
        # AFNI preprocessing.
        self.afni_preprocessor = AFNIPreprocessor(self.params, registration_handler=self.handle_registration_end,\
            baseline_handler=self.hande_baseline_task_end, preprocessing_handler=self.start_dnn_prediction)

        # DNN prediction.
        self.dnn_predictor = TorchCNNPredictor(self.params, prediction_handler=self.manage_probability_queue)

    def write_etime(self, message):
        # 'block_fixation' is repeated in multiple blocks. -> need to skip containment check.
        if not message == 'block_fixation':
            if message in self._etime_message_set:
                return
            else:
                self._etime_message_set.add(message)

        now = datetime.now(tz=timezone('Asia/Seoul'))
        now_formatted = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.params['RUN']['etime_path'], 'a') as f:
            f.write(f'{now_formatted}\t{message}\n')
    
    def unload_resource(self):
        self.afni_preprocessor.quit_preprocessor()
        self.dnn_predictor.quit_dnn_predictor()
    
    def set_sync_status(self, status):
        if status not in [RTConstants.SYNC_ZERO, RTConstants.SYNC_REGISTER, RTConstants.SYNC_BASELINE, RTConstants.SYNC_RT]:
            raise ValueError('Invalid sync status value.')

        if self._sync_status != status:
            self._sync_status = status
    
    def handle_sync(self):
    
        # if it is the first sync signal, 
        # prepare for the registration and annotate it as a "sync" status/event.
        if self._sync_status == RTConstants.SYNC_ZERO:
            self.set_sync_status(RTConstants.SYNC_REGISTER)
            self.write_etime('sync')
        else:
            # for every sync ('s' key press) ...
            super().handle_sync() # increase the current_mri_volume count.


            # if (1) the current sync status is "register",
            if self._sync_status == RTConstants.SYNC_REGISTER:
                # (2) registration is not started, and (3) enough volumes (>= 6) are received,
                # start registration.
                if not self._registration_started and self.current_mri_volume > self.params['RT']['base_vols']:
                    self.start_registration()
            
            # compute a new baseline (right after the registration or in the main task)
            elif self._sync_status == RTConstants.SYNC_BASELINE:
                self.start_baseline_task()
            
            # if the current sync status is 'rt' (currently in the main task),
            # run preprocessing for every sync signal.
            elif self._sync_status == RTConstants.SYNC_RT:
                self.start_preprocessing()
            else:
                raise ValueError('Invalid sync status value.')
    
    def start_registration(self):
        # self.afni_preprocessor.queue_preprocessing_task(registration=True)
        self.afni_preprocessor.queue_preprocessing_task('registration')
        self._registration_started = True
        print(f'DEBUG: starting registration.')
        self.write_etime(f'*registraion_start/vol:{self.current_mri_volume}')
    
    @Slot()
    def handle_registration_end(self):
        self._registration_finished = True
        self.set_sync_status(RTConstants.SYNC_RT)
        # self.set_sync_status(RTConstants.SYNC_BASELINE)
        print(f'DEBUG: finished registration.')
        self.write_etime(f'*registraion_end/vol:{self.current_mri_volume}')
    
    def start_baseline_task(self):
        if self._new_baseline_accept:
            self._baseline_volume = self.current_mri_volume
            self._new_baseline_accept = False
            self.write_etime(f'*baseline_start/vol:{self.current_mri_volume}')

        self.afni_preprocessor.queue_preprocessing_task('baseline', current_volume_index=self.current_mri_volume, \
                                                        baseline_volume_index=self._baseline_volume)
        print(f'DEBUG: queue baseline of volume {self.current_mri_volume}.')

    @Slot()
    def hande_baseline_task_end(self):
        self.set_sync_status(RTConstants.SYNC_RT)
        self.set_preprocessing_on(True) # now start preprocessing
        self._new_baseline_accept = True
        print(f'DEBUG: finished baseline task.')
        self.write_etime(f'*baseline_end/vol:{self.current_mri_volume}/baseline_vol:{self._baseline_volume}')
    
    def start_preprocessing(self):
        if self._preprocessing_on: # run preprocessing only when the flag is on
            self.afni_preprocessor.queue_preprocessing_task('rt', current_volume_index=self.current_mri_volume, \
                                                            baseline_volume_index=self._baseline_volume)
            print(f'DEBUG: queue preprocessing of volume {self.current_mri_volume}.')
    
    @Slot(int)
    def start_dnn_prediction(self, finished_mri_volume):
        if self._dnn_prediction_on:
            self.dnn_predictor.run_prediction(finished_mri_volume)
        else:
            self.dnn_predictor.clear_queue()
    
    @Slot(int, float, float, float)
    def manage_probability_queue(self, predicted_volume, image_prob, sound_prob, text_prob):
        print(f'DEBUG: I: {image_prob}, S: {sound_prob}, L: {text_prob} (volume {predicted_volume})')
        # if len(self._image_probability_queue) == 3: # if the queue contains predictions of previous 3 volumes,
        #     self._image_probability_queue.pop(0) # discard previous probability.
        self._image_probability_queue.append((predicted_volume, image_prob, sound_prob, text_prob)) # insert new probability.
        self.write_etime(f'*dnn_prediction,image={image_prob},sound={sound_prob},text={text_prob}/vol:{predicted_volume}')
    
    def set_dnn_on(self, status: bool):
        self._dnn_prediction_on = status

    def set_preprocessing_on(self, status: bool):
        self._preprocessing_on = status

    def get_probability_queue(self):
        return self._image_probability_queue
    
    def empty_probability_queue(self):
        self._image_probability_queue = []

class RTInstructionPage(TaskPage):
    def __init__(self, parent: TaskMainWindow):
        super().__init__(parent)
        self.task_main_window: RTTaskMainWindow = parent
        params = self.task_main_window.params
        self.delay = params['RT']['ins_duration']

    def init_ui(self):
        self.task_main_window.show_phrase('Run will start after fixation.\nTry to focus on the target.')
    
    def load_resource(self):
        self.waiter = Waiter()
    
    def unload_resource(self):
        self.waiter.quit_waiter()
        self.waiter = None
    
    def load_page(self):
        # self.task_main_window.set_sync_status(RTConstants.SYNC_REGISTER)
        super().load_page()
        self.task_main_window.write_etime(f'instruction')
        self.waiter.wait_and_run(self.delay, self.next_page)

class RTReadyPage(ReadyPage):
    def init_delay(self):
        params = self.task_main_window.params
        delay = params['RT']['ready_duration']
        self.delay = delay
    
    def load_page(self):
        self.task_main_window.write_etime(f'ready')
        super().load_page()

class RTFixationPage(FixationPage):
    def init_delay(self):
        params = self.task_main_window.params
        delay = params['RT']['fix_duration']
        self.delay = delay
    
    def load_page(self):
        self.task_main_window.write_etime(f'fixation_60s')
        super().load_page()

# TODO
class RTCOCOTaskPage(TaskPage):
    def __init__(self, parent: TaskMainWindow):
        super().__init__(parent)
        self.task_main_window: RTTaskMainWindow = parent
        params = self.task_main_window.params
        self.RT = params['RT']
        self.run_order = params['RUN']['run_order']
        self.COCO_IMG_PATH = params['PATH']['coco_img_dir_path']
        self._SELECTED_COCO_SAMPLE_NPZ_PATH = params['PATH']['selected_coco_sample_npz_path']
        self.COCO_SOUND_PATH = params['PATH']['coco_sound_dir_path']

        self.cue_duration = self.RT['cue_duration_per_block']
        self.fixation_duration = self.RT['fixation_duration_per_block']
        self.trial_duration = self.RT['trial_duration']
        self.trial_isi_duration = self.RT['trial_isi_duration']
        self.feedback_duration = self.RT['feedback_duration_per_block']
        self.inter_block_duration = self.RT['inter_block_duration']

        self.n_blocks_per_modality = self.RT['n_blocks_per_modality']
        self.n_trials_per_block = self.RT['n_trials_per_block']

        # new
        _selected_coco_sample_npy = np.load(self._SELECTED_COCO_SAMPLE_NPZ_PATH, allow_pickle=True)
        self.selected_coco_sample_dict = _selected_coco_sample_npy['text_dict'].item()
        _selected_coco_id_list = list(self.selected_coco_sample_dict.keys())
        _n_total_trials = 4 * self.n_blocks_per_modality * self.n_trials_per_block
        self.sampled_coco_id_list = random_sample_coco_id(_selected_coco_id_list, _n_total_trials)
        
        self.SHAM = params['SHAM']
        self.is_sham = self.SHAM is not None
        self.block_target_list = random_shuffle_modalities(self.n_blocks_per_modality) if not self.is_sham \
            else self.SHAM['block_target_list']
        self.block_trial_list = self.sampled_coco_id_list if not self.is_sham else self.SHAM['block_trial_list']
        self.block_feedback_list = None if not self.is_sham else self.SHAM['block_feedback_list']
        
        self.current_block_status = RTConstants.BLOCK_CUE
        self.current_block_target = None
        self.current_trial_count = None
        self.current_block_count = 0

    def init_ui(self):
        pass
    
    def load_resource(self):
        self.waiter = Waiter()
    
    def unload_resource(self):
        self.waiter.quit_waiter()
        self.waiter = None
    
    def center_image(self, coco_id: str):
        coco_img_path = os.path.join(self.COCO_IMG_PATH, f'COCO_train2014_{coco_id:0>12}.jpg')

        coco_img = Image.open(coco_img_path)

        # resize image
        if coco_img.size[0] / coco_img.size[1] >= 800/600:
            new_img_size = (800, int(800 * coco_img.size[1] / coco_img.size[0]))
        else:
            new_img_size = (int(600 * coco_img.size[0] / coco_img.size[1]), 600)
        coco_img_resized = coco_img.resize(new_img_size)

    
        # center the image in 800 x 600
        left_margin = int((800 - new_img_size[0]) / 2)
        top_margin = int((600 - new_img_size[1]) / 2)
        coco_img_centered = Image.new(coco_img.mode, (800, 600), 'black')
        coco_img_centered.paste(coco_img_resized, (left_margin, top_margin))

        return coco_img_centered
    
    def overlay_images_sham(self, image_id, text_id, image_opacity):
        random_babi_image_path = os.path.join(self.BABI_IMG_PATH, f'{image_id}.png')
        random_babi_text_path = os.path.join(self.BABI_IMG_PATH, f'{text_id}.png')

        # PIL version.
        babi_image = Image.open(random_babi_image_path).convert('L')
        babi_text = Image.open(random_babi_text_path).convert('RGBA')

        image_size = babi_image.size
        text_size = babi_text.size

        assert image_size[1] > text_size[1]
        center_height = (image_size[1] - text_size[1]) // 2

        sized_babi_text = Image.new('RGBA', image_size, (255, 255, 255, 0))
        sized_babi_text.paste(babi_text, (0, center_height), babi_text)

        babi_text_array = np.array(sized_babi_text, dtype=np.ubyte)
        mask = (babi_text_array[:, :, :3] == (255, 255, 255)).all(axis=2)
        alpha = np.where(mask, 0, 255)
        babi_text_array[:, :, -1] = alpha
        new_babi_text = Image.fromarray(np.ubyte(babi_text_array)).convert('L')

        blended_image = Image.blend(new_babi_text, babi_image, image_opacity)

        return blended_image, image_id, text_id
    
    def show_block_cue_text(self):
        self.task_main_window.show_phrase('Target: ' + r'$\bf{' + f'{self.current_block_target}' + '}$')

    def show_cross_fixation(self):
        self.task_main_window.show_phrase('+')
    
    def show_trial_stimuli(self, coco_id: str, coco_text: str):
        font_option = {'color': 'white', 'size': 30}

        img = self.center_image(coco_id)
        
        ax = self.task_main_window.set_axis()
        ax.imshow(np.array(img))
        txt = ax.text(400, 500, '\n'.join(textwrap.wrap(coco_text, 30, break_long_words=False, replace_whitespace=False)), fontdict=font_option,\
                horizontalalignment='center', verticalalignment='center')
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
        self.task_main_window.canvas.draw()
    
    def play_coco_sound(self, coco_id: str):
        coco_sound_path = os.path.join(self.COCO_SOUND_PATH, f'{coco_id:0>12}.mp3')
        coco_sound = pygame.mixer.Sound(coco_sound_path)
        coco_sound_length = coco_sound.get_length()
        coco_sound.play()

        return coco_sound_length

    def manage_task_block(self):
        self.waiter.quit_waiter()
        self.task_main_window.fig.tight_layout(pad=0)
        self.task_main_window.clear_figure()

        if self.current_block_status == RTConstants.BLOCK_CUE:
            self.task_main_window.set_dnn_on(False)
            self.task_main_window.empty_probability_queue()

            self.current_block_target = self.block_target_list.pop(0)
            self.current_block_count += 1
            print(f'[block {self.current_block_count}] target = {self.current_block_target}')
            self.show_block_cue_text()
            self.task_main_window.write_etime(f'block_{self.current_block_count}_cue,target:{self.current_block_target}')

            self.current_block_status = RTConstants.BLOCK_FIXATION
            self.waiter.wait_and_run(self.cue_duration, self.manage_task_block)

        elif self.current_block_status == RTConstants.BLOCK_FIXATION:
            self.show_cross_fixation()
            self.task_main_window.set_sync_status(RTConstants.SYNC_BASELINE)
            self.task_main_window.write_etime(f'block_{self.current_block_count}_fixation')

            self.current_block_status = RTConstants.BLOCK_TRIAL
            self.current_trial_count = 0
            self.waiter.wait_and_run(self.fixation_duration, self.manage_task_block)

        elif self.current_block_status == RTConstants.BLOCK_TRIAL:
            self.task_main_window.set_dnn_on(True)
            self.current_trial_count += 1

            # if not self.is_sham:
            #     # sound_length = self.show_trial_stimuli()
            #     coco_image_id, coco_sound_id, coco_text_id = self.sampled_coco_id_list.pop(0)
            # else:
            #     # sound_length = self.show_trial_stimuli_sham()
            #     coco_image_id, coco_sound_id, coco_text_id = self.sampled_coco_id_list.pop(0)
            coco_image_id, coco_sound_id, coco_text_id = self.block_trial_list.pop(0)
            print('image_id', coco_image_id, 'sound_id', coco_sound_id, 'text_id', coco_text_id)
            
            coco_text = self.selected_coco_sample_dict[coco_text_id]
            self.show_trial_stimuli(coco_image_id, coco_text)
            coco_sound_length = self.play_coco_sound(coco_sound_id)

            assert coco_text_id in self.selected_coco_sample_dict

            print(f'[trial {self.current_trial_count} (volume {self.task_main_window.current_mri_volume})]')
            self.task_main_window.write_etime(f'block_{self.current_block_count}_trial_{self.current_trial_count},image:{coco_image_id},sound:{coco_sound_id},text:{coco_text_id},sound_length:{coco_sound_length}/volume:{self.task_main_window.current_mri_volume}')
            
            self.current_block_status = RTConstants.BLOCK_ISI
            self.waiter.wait_and_run(coco_sound_length, self.manage_task_block)
        
        elif self.current_block_status == RTConstants.BLOCK_ISI:
            self.show_cross_fixation()

            if self.current_trial_count < self.n_trials_per_block:
                self.task_main_window.write_etime(f'block_{self.current_block_count}_trial_{self.current_trial_count}_isi')
                self.current_block_status = RTConstants.BLOCK_TRIAL
                self.waiter.wait_and_run(self.trial_isi_duration, self.manage_task_block)
            else:
                # if this is the last trial, wait for the inter-block duraiton
                self.task_main_window.write_etime(f'block_{self.current_block_count}_ibi')
                self.task_main_window.set_dnn_on(False)

                if len(self.block_target_list) > 0:
                    self.current_block_status = RTConstants.BLOCK_CUE
                    self.waiter.wait_and_run(self.inter_block_duration, self.manage_task_block)
                elif len(self.block_target_list) == 0:
                    self.waiter.wait_and_run(self.inter_block_duration, self.next_page)

        else:
            raise ValueError('Invalid block status value.')
    
    def load_page(self):
        super().load_page()
        self.manage_task_block()
    
    def unload_page(self):
        super().unload_page()

class RTEndPage(EndPage):
    def load_page(self):
        self.task_main_window.unload_resource()
        self.task_main_window.write_etime(f'end')
        super().load_page()
