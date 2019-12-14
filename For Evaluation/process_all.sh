#!/bin/bash

for i in {0..3..1}
do
	python3 ../transform_test.py $i 
	python3 run_model.py --checkpoint checkpoint_with_model.pt --scene_graphs_json val_sg.json 
	python3 condi_cp.py $i
	ls -l new_outputs/ | wc -l
	ls -l outputs/ | wc -l
	rm ../val_sg.json
	rm outputs/*
done

