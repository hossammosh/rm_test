import os
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    base_dir = os.path.join(os.getcwd(), 'data')
    
    # Project directory
    settings.prj_dir = os.getcwd()
    settings.save_dir = os.path.join(os.getcwd(), 'output')
    
    # Set checkpoint directory to the correct location
    settings.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    
    # LaSOT dataset path
    settings.lasot_path = os.path.join(base_dir, 'lasot')
    
    # Network and results paths
    settings.network_path = os.path.join(os.getcwd(), 'lib', 'test', 'networks')
    settings.result_plot_path = os.path.join(os.getcwd(), 'lib', 'test', 'result_plots')
    settings.results_path = os.path.join(os.getcwd(), 'lib', 'test', 'tracking_results')
    
    return settings
