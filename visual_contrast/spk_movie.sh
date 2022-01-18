#!/bin/bash -l
#SBATCH --job-name=spk_movie
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load unstable py-bbp-analysis-framework
source /gpfs/bbp.cscs.ch/home/pokorny/ConntilityKernel/bin/activate

python -u /gpfs/bbp.cscs.ch/home/pokorny/JupyterLab/git/sscx-analysis/spk_movie/spk-movie-workflow.py simulations.pkl spk_movie_input_cfg.json
python -u /gpfs/bbp.cscs.ch/home/pokorny/JupyterLab/git/sscx-analysis/spk_movie/spk-movie-workflow.py simulations.pkl spk_movie_cfg.json

# EXAMPLE HOW TO RUN: sbatch spk_movie.sh

# CONVERT .png frames to .mp4 @ 5Hz (x10), 25 Hz (2x), or 50 Hz (=realtime):
# /gpfs/bbp.cscs.ch/home/pokorny/Tools/ffmpeg-git-20211123-amd64-static/ffmpeg -r 5 -i frame%04d.png -vcodec h264 -pix_fmt yuv420p -crf 22 spk_movie_x10.mp4
# /gpfs/bbp.cscs.ch/home/pokorny/Tools/ffmpeg-git-20211123-amd64-static/ffmpeg -r 25 -i frame%04d.png -vcodec h264 -pix_fmt yuv420p -crf 22 spk_movie_x2.mp4
# /gpfs/bbp.cscs.ch/home/pokorny/Tools/ffmpeg-git-20211123-amd64-static/ffmpeg -r 50 -i frame%04d.png -vcodec h264 -pix_fmt yuv420p -crf 22 spk_movie_x1.mp4