import os
import glob
import logging

#all checkpoints are folders with sub-directory "checkpoint"
CHECKPOINT_SUBDIR_NAME="checkpoint"
logger = logging.getLogger()

def filter_out_by_keyword(to_filter, keywords):
    logger.info(f"Filtering out checkpoints by keywords {keywords}")
    return [x for x in to_filter if all([y not in x for y in keywords])]


def get_candidate_checkpoint_paths(base_filepath):
    return [x.split("checkpoint")[0] for x in glob.glob(f'{base_filepath}/**/*/checkpoint/')]
    



def get_folder_timestamp_from_file(folder):
    """Given a checkpoint folder, use a specified file 
    to estimate a folder-level timestamp
    
    """

    subfolder = os.path.join(folder,CHECKPOINT_SUBDIR_NAME)
    possible_files_for_timestamp = [os.path.join(subfolder,x) for x in os.listdir(subfolder)]

    #check they are all actually files
    possible_files_for_timestamp = [x for x in possible_files_for_timestamp if os.path.isfile(x)]

    #no file preference
    file_for_timestamp = possible_files_for_timestamp[0]

    mod_time = os.path.getmtime(file_for_timestamp)

    return mod_time 


def get_most_recent_checkpoint(base_filepath, filter_keywords=None):
    candidate_checkpoints = get_candidate_checkpoint_paths(base_filepath)
    
    #filter out by keyword
    if filter_keywords:
        logger.info(f"Filtering out checkpoints with keywords {filter_keywords}")
        candidate_checkpoints = filter(lambda x: all([y not in x for y in filter_keywords]),candidate_checkpoints)
        #candidate_checkpoints = filter_out_by_keyword(candidate_checkpoints, filter_keywords)

    candidate_checkpoints_with_timestamps = [{"folder":x,"timestamp":get_folder_timestamp_from_file(x)} for x in candidate_checkpoints]   

    #sort by timestamp, get most recent
    sorted_candidate_checkpoints = sorted(candidate_checkpoints_with_timestamps, key = lambda x:x["timestamp"])
    selected_checkpoint = sorted_candidate_checkpoints[-1]["folder"]

    logger.info(f"Selecting checkpoint {selected_checkpoint} based on timestamp recency")

    return selected_checkpoint 
    
def latest_ckpt_wrapper_from_cfg(cfg):

    base_filepath = os.path.join(cfg.trainer.default_root_dir,cfg.run_name)
    logger.info(f"Finding latest checkpoint in {base_filepath}")

    if hasattr(cfg,"filter_keywords"):
        logger.info("Filtering out checkpoints with keywords {filter_keywords}")    
        filter_keywords = cfg.filter_keywords
    else:
        logger.info("Not filtering checkpoints by keyword")
        filter_keywords = None
    
    latest_checkpoint = get_most_recent_checkpoint(base_filepath, filter_keywords=filter_keywords)
    logger.info(f"Determined latest checkpoint to be {latest_checkpoint}")
    return latest_checkpoint


def main():
    """
    Assumes checkpoint names take the format USER/EXPERIMENT_NAME/RUN_NAME/TIMESTAMP/CKPT_FOLDER/
    """   
    
    logging.basicConfig(level=logging.INFO)
 
    USER="colehawk"
    EXPERIMENT_NAME="easel"
    #RUN_NAME="patched_20B_stage_2_t5"
    RUN_NAME="20B_stage_2_t5"
    FILTER_KEYWORDS=["last.ckpt"]

    base_filepath = f"/mnt_out/{USER}/{EXPERIMENT_NAME}/{RUN_NAME}"

    most_recent_checkpoint = get_most_recent_checkpoint(base_filepath, filter_keywords=FILTER_KEYWORDS)


if __name__=='__main__':
    main()
