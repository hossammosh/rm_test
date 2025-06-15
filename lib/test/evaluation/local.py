from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/e/current_research/seq/data/got10k_lmdb'
    settings.got10k_path = '/mnt/e/current_research/seq/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/mnt/e/current_research/seq/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/mnt/e/current_research/seq/data/lasot_lmdb'
    settings.lasot_path = '/mnt/e/current_research/seq/data/lasot'
    settings.network_path = '/mnt/e/current_research/seq/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/e/current_research/seq/data/nfs'
    settings.otb_path = '/mnt/e/current_research/seq/data/OTB2015'
    settings.prj_dir = '/mnt/e/current_research/seq'
    settings.result_plot_path = '/mnt/e/current_research/seq/test/result_plots'
    settings.results_path = '/mnt/e/current_research/seq/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/e/current_research/seq'
    settings.segmentation_path = '/mnt/e/current_research/seq/test/segmentation_results'
    settings.tc128_path = '/mnt/e/current_research/seq/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/e/current_research/seq/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/e/current_research/seq/data/trackingnet'
    settings.uav_path = '/mnt/e/current_research/seq/data/UAV123'
    settings.vot_path = '/mnt/e/current_research/seq/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

