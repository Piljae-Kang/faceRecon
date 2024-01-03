import argparse
import os
import subprocess

def parse_args():

    parser = argparse.ArgumentParser ( prog = "colmap helper", description = "")
    
    # arguments 
    parser.add_argument ('-ipath', dest ="image_path", action = 'store')
    parser.add_argument ('-dpath', dest ="data_path", action = 'store')
    parser.add_argument ('-start', dest ="start", action = 'store', type = int, default=0 )
    parser.add_argument ('-end', dest ="end", action = 'store', type = int, default=7 ) 

    # get args
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    # 0 feature extractor
    # 1 exhaustive matcher
    # 2 colmap mapper
    # 4 undistorter
    # 5 stereo matching
    # 6 stereo fusion
    # 7 poisson reconstruction
    # 8 delaunay mesher

    command_template_list = [
    # feature extractor [0]
    "colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $IMAGE_PATH",
    # exhaustive matcher [1]
    "colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db",
    "mkdir $DATASET_PATH/sparse",
    # colmap mapper [2]
    "colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $IMAGE_PATH \
    --output_path $DATASET_PATH/sparse",
    "mkdir $DATASET_PATH/dense",
    # undistorter [3]
    "colmap image_undistorter \
    --image_path $IMAGE_PATH \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000",
    # stereo matching [4]
    "colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true",
    # stereo fusion [5]
    "colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply",
    # poisson reconstruction [6]
    "colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply",
    # delaunay mesher [7]
    "colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply"]    

    # update colmap command
    command_template_list = [ command.split() for command in command_template_list ]
    commands = [ [token.replace ("$DATASET_PATH", args.data_path) for token in command] for command in command_template_list ]
    commands = [ [token.replace ("$IMAGE_PATH", args.image_path) for token in command] for command in commands ]
    
    print ( commands )

    # run commands
    for idx, command in enumerate (commands[args.start:args.end]):

        print( idx, " ", command)

        subprocess.run ( command)
        #subprocess.