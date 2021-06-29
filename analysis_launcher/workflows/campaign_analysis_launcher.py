# Description: BBP-WORKFLOW code for launching arbitrary simulation campaign analyses
# Author: C. Pokorny
# Created: 28/06/2021

import os
import numpy as np
from luigi import ListParameter, DictParameter, WrapperTask
from bbp_workflow.simulation import LookupSimulationCampaign
from bbp_workflow.utils import xr_from_dict
from bbp_workflow.luigi import RunAnywayTarget
from bbp_workflow.task import SbatchTask

""" Campaign analysis launcher (wrapper task), launching separate analysis tasks as specified """
class CampaignAnalysisLauncher(WrapperTask):
    
    analyses = ListParameter(default=[])
    
    def requires(self):
        
        # Load simulation campaign config from Nexus URL
        sim_campaign_cfg = LookupSimulationCampaign().output().entity
        config = xr_from_dict(sim_campaign_cfg.configuration.as_dict())  # Get sim campaign config as Xarray
        root_path = os.path.join(config.attrs['path_prefix'], config.name) # Root path of simulation campaign
        sim_paths = config.to_series() # Single simulation paths as Pandas series with multi-index
        assert os.path.commonpath(sim_paths.tolist()) == root_path, 'ERROR: Root path mismatch!'
        
        print(f'\nINFO: Loaded simulation campaign "{sim_campaign_cfg.name}" from {sim_campaign_cfg.get_url()} with coordinates {list(sim_paths.index.names)}\n')
        
        # Check if simulation results exist
        valid_sims = [os.path.exists(os.path.join(p, 'out.dat')) for p in sim_paths]
        sim_paths = sim_paths[valid_sims]
        
        print(f'\nINFO: Found {np.sum(valid_sims)} of {len(valid_sims)} completed simulations\n')
        
        # Create simulation paths to BlueConfigs
        sims = sim_paths.apply(lambda p: os.path.join(p, 'BlueConfig'))
        assert np.all([os.path.exists(s) for s in sims.values.flatten()]), 'ERROR: BlueConfig(s) missing!'
        
        # Write simulation file to analyses folder
        launch_path = os.path.join(root_path, 'analyses')
        if not os.path.exists(launch_path):
            os.makedirs(launch_path)
        
        sims.to_pickle(os.path.join(launch_path, 'simulations.pkl'))
        
        # Prepare & launch analyses, as specified in launch config
        num_analyses = len(self.analyses)
        print(f'INFO: Launching {num_analyses} analyses...')
        
        # TODO: Write config files
        #       Clone scripts
        #       Launch scripts
        
        return [] #self.analyses()
    
#     def analyses(self):
        
#         num_analyses = len(self.analyses)
#         print(f'RUNNING {num_analyses} analyses...')
        
#         return [CampaignAnalysis(analysis_params={'name': f'test{n}'}) for n in range(num_analyses)]


""" Campaign analysis task, running an analysis as SLURM job """
class CampaignAnalysis(SbatchTask):
    
    analysis_params = DictParameter(default={})
    command = ''
    
    def run(self):
        print(f'RUNNING Analysis with params: {self.analysis_params}')
        
        self.command = 'python -u payload.py'
        self.args = ''
        self.chdir = '/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/bbp-workflow/workflows/testing/workflows'
        self.job_name = self.analysis_params.get('name', 'unknown') + '_analysis'
        self.module_archive = 'unstable'
        self.modules = 'python' # Loading latest Python module
        
        super().run()
        
        self.output().done()
    
    def output(self):
        return RunAnywayTarget(self)

