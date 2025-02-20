datagen: octo_datagen.py
	python3 octo_datagen.py

train_model: octo_model.py
	python3 octo_model.py

viz: octo_viz.py ./simulator
	python3 octo_viz.py -m simulator.simutil

util: util.py
	python3 util.py