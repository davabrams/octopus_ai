all: generate_octopus generate_surface generate_agent

generate_octopus: octopus_generator.py
	python3 octopus_generator.py -m surface_generator agent_generator simutil

generate_surface: octopus_generator.py
	python3 surface_generator.py -m surface_generator agent_generator simutil

generate_agent: octopus_generator.py
	python3 agent_generator.py -m surface_generator agent_generator simutil