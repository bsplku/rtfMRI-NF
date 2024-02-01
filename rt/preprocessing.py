import time
import os
from nipype.interfaces import afni 

class Preprocessing:
    def __init__(self, params, volume_number='base', name_base=None):
        os.environ['NO_ET'] = 'True'
        self.params = params
        self.raw_path = self.params['RUN']['subject_raw_path']
        self.series = self.params['RUN']['mri_series']
        self.series_path = self.params['RUN']['subject_series_path']
        
        self.volume_number = volume_number
        # self.name_base = name_base if name_base is not None else 'epi'
        self.name_base = 'epi'
        
        # if self.volume_number == 'base' or self.volume_number <= 5:
        if self.volume_number == 'base':
            # for registration; 000001.dcm - 000005.dcm (initial 5 volumes) -> base.nii
            self.in_file = ' '.join(['%s/001_%s_%s.dcm' % (self.raw_path, str(self.series).zfill(6), str(dcm_num).zfill(6)) for dcm_num in range(1, 6)])
            self.out_file = '%s_%s_%s.nii' % (self.name_base, str(self.series).zfill(3), self.volume_number)
        else :
            # for normal preprocessing; one .dcm volume to one .nii file
            self.in_file = '%s/001_%s_%s.dcm' % (self.raw_path, str(self.series).zfill(6), str(self.volume_number).zfill(6))
            self.out_file = '%s_%s_%s.nii' % (self.name_base, str(self.series).zfill(3), str(self.volume_number).zfill(3))
            
        
    def __str__(self):
        return self.command.cmdline
    
    def update_file_name(self, next_step=None, custom_file=None):
        if next_step is None and custom_file is None:
            self.in_file = self.out_file
        
        if next_step is not None and custom_file is None:
            self.in_file = self.out_file
            self.out_file = self.in_file.split('.')[:-1][-1] + '_%s.nii' % next_step
            
        if next_step is not None and custom_file is not None:
            self.out_file = custom_file.split('.')[:-1][-1] + '_%s.nii' % next_step
            self.in_file = custom_file
        
        if next_step is None and custom_file is not None :
            self.out_file = custom_file
            self.in_file = custom_file
            
    
    def run(self):
        self.command.run()

    
    def copy(self, in_file, out_file):
        '''
        should specify both in_file and out_file parameters.
        '''

        series_path = self.params['RUN']['subject_series_path']
        # base_file_name = out_file if out_file is not None else in_file.split('.')[:-1][-1]
        # create_out_file = out_file if out_file is not None else '%s.nii' % base_file_name
        
        if '/' in in_file:
            # if '/' is in in_file parameter (i.e., full path), do not concetenate in_file and series_path
            self.command = afni.base.CommandLine(command='3dcopy %s %s/%s' % (in_file, series_path, out_file))        
        else :
            self.command = afni.base.CommandLine(command='3dcopy %s/%s %s/%s' % (series_path, in_file, series_path, out_file))        
    
    
    def to3d(self):
        tr = self.params['MRI']['tr'] * 1000
        slices = self.params['MRI']['slices']     
        volume = (5 if self.volume_number == 'base' else 1)

        first_volume_path = '%s/001_%s_%s.dcm' % (self.raw_path, str(self.series).zfill(6), '1'.zfill(6))
        first_volume_size = os.path.getsize(first_volume_path)
        
        # wait until the current volume has the same or similar size of the first volume.
        while True and volume == 1:
            if not os.path.exists(self.in_file):
                time.sleep(0.01)
                continue
            this_volume_size = os.path.getsize(self.in_file)
            size_diff = this_volume_size - first_volume_size
            if size_diff < 0:
                time.sleep(0.01)
            else:
                break
        
        self.command = afni.base.CommandLine(command ='to3d -time:zt %s %s %s alt+z -session %s \
                                            -overwrite -datum float -epan -prefix %s %s' % 
                                     (
                                         slices, volume, tr, self.series_path, 
                                         self.out_file, self.in_file
                                     ), 
                                     environ={'DISPLAY': ':1'})
            
        self.update_file_name() # out_file (the converted .nii file) becomes the in_file
    

    def align_epi_anat(self):       
        self.command = afni.AlignEpiAnatPy()
        self.command.inputs.anat = '%s/MNI_EPI.nii' % self.series_path
        self.command.inputs.in_file = '%s/%s' % (self.series_path, self.in_file)
        self.command.inputs.epi_base = 0
        self.command.inputs.epi_strip = '3dAutomask'
        self.command.inputs.volreg = 'off'
        self.command.inputs.tshift = 'off'
        self.command.inputs.epi2anat = True
        self.command.inputs.save_skullstrip = False
        self.command.inputs.suffix = '_al'
        self.command.inputs.args = '-overwrite -cmass cmass -anat_has_skull no -resample on -cost lpa -multi_cost ls nmi -align_centers on -giant_move -output_dir %s' % self.series_path
        
        # next step will be transform the tlrc file (base_al+tlrc; registration output) to nii file (base_al.nii)
        # no need to update the filename
        # self.update_file_name(next_step='al')
    

    def alignment(self):
        self.command = afni.Allineate()
        self.command.inputs.in_file = '%s/%s' % (self.series_path, self.in_file) 
        self.command.inputs.out_file = '%s/%s' % (self.series_path, self.out_file) # in_file and out_file are same here (epi_[series]_[volume].nii)
        self.command.inputs.in_matrix = '%s/epi_%s_base_al_mat.aff12.1D' % (self.series_path, str(self.series).zfill(3))
        self.command.inputs.master = '%s/epi_%s_base_al.nii' % (self.series_path, str(self.series).zfill(3))
        self.command.inputs.args = '-final NN -overwrite' # overwrite the to3d'd epi_[series]_[volume].nii file
        
        self.update_file_name(next_step='volreg') # self.in_file = epi_[series]_[volume].nii, self.out_file = epi_[series]_[volume]_volreg.nii
    

    def volreg(self):
        self.command = afni.Volreg()
        self.command.inputs.in_file = '%s/%s' % (self.series_path, self.in_file) # in_file = epi_[series]_[volume].nii
        self.command.inputs.oned_file = '%s/epi_%s_mot.1D' % (self.series_path, str(self.series).zfill(3))
        self.command.inputs.out_file = '%s/%s' % (self.series_path, self.out_file) # out_file = epi_[series]_[volume]_volreg.nii
        self.command.inputs.basefile = '%s/%s_%s_base_al.nii' % (self.series_path, self.name_base, str(self.series).zfill(3))
        self.command.inputs.args = '-nomaxdisp -cublic -overwrite'
        
        self.update_file_name(next_step='blur') # self.in_file = epi_[series]_[volume]_volreg.nii, self.out_file = epi_[series]_[volume]_volreg_merge.nii


    def blur(self):
        self.command = afni.Merge()
        self.command.inputs.in_files = '%s/%s' % (self.series_path, self.in_file) # in_file = epi_[series]_[volume]_volreg.nii
        self.command.inputs.blurfwhm = 8
        self.command.inputs.doall = True
        self.command.inputs.outputtype = 'NIFTI'
        self.command.inputs.out_file = '%s/%s' % (self.series_path, self.out_file)
        self.command.inputs.args = '-overwrite' # out_file = epi_[series]_[volume]_volreg_blur.nii
        # self.command.inputs.args = '-overwrite -session %s -prefix %s' % (self.series_path, self.out_file) # out_file = epi_[series]_[volume]_volreg_blur.nii
    
        self.update_file_name() # self.in_file = epi_[series]_[volume]_volreg_blur.nii, self.out_file = epi_[series]_[volume]_volreg_blur.nii


    # def tcat(self):
    #     file = '%s_%s_tcat.nii' % (self.name_base, str(self.series).zfill(3))
    #     base_file = os.path.join(self.series_path, file)
        
    #     if not os.path.exists(base_file):
    #         self.copy(self.in_file, file)
    #     else:
    #         self.command = afni.TCat()
    #         self.command.inputs.in_files = ['%s/%s_%s_tcat.nii' % (self.series_path, self.name_base, str(self.series).zfill(3)), os.path.join(self.series_path, self.in_file)]
    #         self.command.inputs.out_file = base_file # '%s/%s' % (self.series_path, base_file)
    #         self.command.inputs.rlt = '+'
    #         self.command.inputs.args = '-overwrite'
        
    #     self.update_file_name(custom_file=file)

    def tcat(self, baseline_volume):
        '''
        need to specify baseline_volume
        '''

        baseline_name = '%s_%s_%s_base.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3))
        baseline_path = os.path.join(self.series_path, baseline_name)
        
        if not os.path.exists(baseline_path):
            self.copy(self.in_file, baseline_name)
        else:
            self.command = afni.TCat()
            self.command.inputs.in_files = [baseline_path, os.path.join(self.series_path, self.in_file)]
            self.command.inputs.out_file = baseline_name # '%s/%s' % (self.series_path, base_file)
            self.command.inputs.rlt = '+'
            self.command.inputs.args = '-overwrite'
        
        # self.update_file_name(custom_file=file)
    
    def mean(self, baseline_volume):
        '''
        need to specify baseline_volume
        '''

        baseline_name = '%s_%s_%s_base.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3))

        self.command = afni.TStat()
        self.command.inputs.in_file = '%s' % baseline_name
        self.command.inputs.args = '-mean'
        self.command.inputs.out_file = '%s_%s_%s_base_mean.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3)) # 'rm.mean.epi_%s.nii' % str(self.series).zfill(3)
        self.command.inputs.args = '-overwrite'
    

    # def automask(self, baseline_volume):
    #     '''
    #     need to specify baseline_volume
    #     '''

    #     baseline_name = '%s_%s_%s_base.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3))

    #     self.command = afni.Automask()
    #     self.command.inputs.in_file = '%s/%s' % (self.series_path, baseline_name)
    #     self.command.inputs.outputtype = 'NIFTI'
    #     self.command.inputs.dilate = 1
    #     self.command.inputs.out_file = '%s_%s_%s_base_mask.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3)) # 'rm.mean.epi_%s.nii' % str(self.series).zfill(3) # 'rm.mask.epi_%s.nii' % (str(self.series).zfill(3))
    #     self.command.inputs.args = '-overwrite'
        

    def scale(self, baseline_volume):
        '''
        need to specify baseline_volume
        '''

        baseline_mean_name = '%s_%s_%s_base_mean.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3))
        baseline_mask_name = '%s_%s_%s_base_mask.nii' % (self.name_base, str(self.series).zfill(3), str(baseline_volume).zfill(3))

        self.command = afni.Calc()
        self.command.inputs.in_file_a = '%s/%s' % (self.series_path, baseline_mean_name)
        self.command.inputs.in_file_b = '%s/%s' % (self.series_path, self.in_file)
        # self.command.inputs.in_file_c = '%s/%s' % (self.series_path, baseline_mask_name)
        self.command.inputs.expr = 'max(min((b-a)/a*100, 10), -10)' # 'step(c)*max(min((b-a)/a*100, 10), -10)'
        self.command.inputs.out_file = '%s/epi_%s_%s_scale.nii' % (self.series_path, str(self.series).zfill(3), str(self.volume_number).zfill(3))
        self.command.inputs.args = '-overwrite'

    

    
