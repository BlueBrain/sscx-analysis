# Description: BBP-WORKFLOW code for launching arbitrary simulation campaign analyses
# Author: C. Pokorny
# Created: 28/06/2021

import os
import json
import subprocess
import numpy as np
from luigi import Task, Parameter, ListParameter, DictParameter
from bbp_workflow.simulation import LookupSimulationCampaign
from bbp_workflow.utils import xr_from_dict
from bbp_workflow.luigi import RunAnywayTarget
from bbp_workflow.task import SbatchTask

""" Campaign analysis launcher, preparing config files and launching separate analysis tasks as specified """
class CampaignAnalysisLauncher(Task):
    
    list_of_analyses = ListParameter(default=[])
    
    def requires(self):
        return LookupSimulationCampaign()
    
    def run(self):
        
        # Load simulation campaign config from Nexus URL
        sim_campaign_cfg = sim_campaign_cfg = self.input().entity
        config = xr_from_dict(sim_campaign_cfg.configuration.as_dict())  # Get sim campaign config as Xarray
        root_path = os.path.join(config.attrs['path_prefix'], config.name) # Root path of simulation campaign
        sim_paths = config.to_series() # Single simulation paths as Pandas series with multi-index
        assert os.path.commonpath(sim_paths.tolist()) == root_path, 'ERROR: Root path mismatch!'
        
        print(f'\nINFO: Loaded simulation campaign "{sim_campaign_cfg.name}" from {sim_campaign_cfg.get_url()} with coordinates {list(sim_paths.index.names)}')
        
        # Check if simulation results exist
        valid_sims = [os.path.exists(os.path.join(p, 'out.dat')) for p in sim_paths]
        sim_paths = sim_paths[valid_sims]
        
        print(f'INFO: Found {np.sum(valid_sims)} of {len(valid_sims)} completed simulations to analyze')
        
        # Create simulation paths to BlueConfigs
        sims = sim_paths.apply(lambda p: os.path.join(p, 'BlueConfig'))
        assert np.all([os.path.exists(s) for s in sims.values.flatten()]), 'ERROR: BlueConfig(s) missing!'
        
        # Write simulation file to analyses folder
        launch_path = os.path.join(root_path, 'analyses')
        if not os.path.exists(launch_path):
            os.makedirs(launch_path)
        sim_file = 'simulations.pkl'
        sims.to_pickle(os.path.join(launch_path, sim_file))
        
        # Prepare & launch analyses, as specified in launch config
        num_analyses = len(self.list_of_analyses)
        print(f'INFO: {num_analyses} campaign analyses to launch: {[anlys["name"] for anlys in self.list_of_analyses]}\n')
        
        analysis_tasks = []
        for anlys in self.list_of_analyses:
            anlys_name = anlys['name']
            anlys_script = anlys['script']
            anlys_params = dict(anlys['parameters'])
            anlys_res = anlys['resources']
            
            # Create script folder
            script_path = os.path.join(launch_path, 'scripts', anlys_name)
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            
            # Write parameters
            anlys_params['out_root'] = script_path
            param_file = 'parameters.json'
            with open(os.path.join(script_path, param_file), 'w') as f:
                json.dump(anlys_params, f, indent=2)
            
            # Download script from GIT repository to script_path
            script_name = os.path.split(anlys_script)[-1]
            script_file = os.path.join(script_path, script_name)
            # TODO...
#             if os.path.isfile(script_file):
#                 os.remove(script_file) # Remove if already exists
#             proc = subprocess.Popen(f'wget -P {script_path} {anlys_script}', shell=True, stdout=subprocess.PIPE)
#             print(proc.communicate()[0].decode())
            
            # Prepare tasks
            cmd = f'python -u {script_name}'
            args = f'{os.path.join(os.path.relpath(launch_path, script_path), sim_file)} {param_file}'
            module_archive = anlys.get('module_archive', 'unstable')
            modules = anlys.get('modules', 'python') # Loading latest Python module by default
            anlys_res = {k: str(v) for k, v in anlys_res.items()} # Convert values to str, to avoid warning from parameter parser when directly passing whole "resources" dict
            analysis_tasks.append(CampaignAnalysis(name=anlys_name, chdir=script_path, command=cmd, args=args, module_archive=module_archive, modules=modules, **anlys_res))
        
        yield analysis_tasks # Launch tasks
        
        self.output().done()
    
    def output(self):
        return RunAnywayTarget(self)


""" Campaign analysis task, running an analysis as SLURM job """
class CampaignAnalysis(SbatchTask):
    
    name = Parameter()
    
    def run(self):
        print(f'\nINFO: Running campaign analysis task "{self.name}"\n')
        print(f'Command: {self.command} {self.args}') # [TESTING]
        print(f'Modules: {self.module_archive} {self.modules}') # [TESTING]
        
        self.job_name = 'CampaignAnalysis[' + self.name + ']'
        
        #super().run()
        
        self.output().done()
    
    def output(self):
        return RunAnywayTarget(self)
