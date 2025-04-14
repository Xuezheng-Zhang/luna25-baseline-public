BATCH --partition=csedu
#SBATCH --account=cseduproject # or your course code
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --output=my-experiment-%j.out
#SBATCH --error=my-experiment-%j.err
#SBATCH --mail-user=YOURNAME
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Hello! This very simple experiment works."

