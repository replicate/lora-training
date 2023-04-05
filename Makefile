.PHONY: cog-yaml-basic
cog-yaml-basic:
	PREDICT_FILE=predict_basic.py envsubst < cog.yaml.in > cog.yaml

.PHONY: cog-yaml-advanced
cog-yaml-advanced:
	PREDICT_FILE=predict_advanced.py envsubst < cog.yaml.in > cog.yaml

.PHONY: push-basic
push-basic: cog-yaml-basic
	cog push r8.im/votiethuy/lora-training

.PHONY: push-advanced
push-advanced: cog-yaml-advanced
	cog push r8.im/votiethuy/lora-advanced-training

.PHONY: push
push: lint push-basic push-advanced

.PHONY: lint
lint: cog-yaml-basic
	cog run flake8 *.py
