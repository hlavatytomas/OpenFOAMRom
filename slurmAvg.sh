#!/bin/bash
# Name of the job
#SBATCH -J pyAnalysis
# Partition to use
#SBATCH -p Mshort
# Time limit. Often not needed as there are defaults,
# but you might have to specify it to get the maximum allowed.
# time: 24hours
#SBATCH --time=2-00:00:00
# Number of processes
## SBATCH -n96
#SBATCH -n1
# Do NOT use some nodes
##SBATCH --exclude=kraken-m[1-7,9]
##SBATCH --nodelist=kraken-m9
##sbatch --nodes=1

# run testForAda on kraken
# python3 -u testForAda.py > log.testForAda
# python3 -u seeOdtrzeni.py > log.seeOdtrzeni
# python3 -u testMyROMV1.py > logs/log.hor230828
python3 -u testMyROMV1.py > logs/log.ver230828
# python3 -u testMyROMV1.py > log.avgTopVer