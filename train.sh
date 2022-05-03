# The training process may collapse, use this to restart training automatically
for i in $(seq 1 100)
do
	python main.py
#  python -u train1gpu.py >> work_dir/govsr_4+2_56_1gpu.log 2>&1 &
#  (python main.py >> work_dir/dis1gpu/govsr_4+2_56_dis1gpu.log 2>&1 &)
	sleep 5
done