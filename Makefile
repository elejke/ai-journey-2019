###### START CHANGABLE ######

# path to the code
BASE_PATH     := .
# path to the data
DATA_PATH     := ./data/check

###### END CHANGABLE ######

ABS_BASE_PATH := $$(realpath .)/${BASE_PATH}
IMAGE         := $$(jq -r ".image" ${BASE_PATH}/metadata.json)
RUNNER        := $$(jq -r ".entry_point" ${BASE_PATH}/metadata.json)

all:
	@echo "Please specify target"

predict: create predictor destroy
	@echo "$@ is done!"

evaluate: create evaluator destroy
	@echo "$@ is done!"

create:
	sudo docker run \
		-d \
		-v ${ABS_BASE_PATH}/src:/root/solution/src \
		-v ${ABS_BASE_PATH}/models:/root/solution/models \
		-p 8000:8000 \
		--memory="16g" \
		--memory-swap="16g" \
		--cpus="4" \
		--name="tester" \
		${IMAGE} \
		/bin/bash -c "cd /root/solution && ${RUNNER}"

predictor:
	cd client && \
	python predictor.py --folder-path ../${DATA_PATH} --url http://localhost:8000 && \
	cd ..

evaluator:
	cd client && \
	python evaluator.py --folder-path ../${DATA_PATH} --url http://localhost:8000 && \
	cd ..

destroy:
	sudo docker stop tester
	sudo docker rm tester

submit:
	cd ${ABS_BASE_PATH} && \
	zip -r src.zip src models metadata.json -x *__pycache__* && \
	cd - && \
	mv ${ABS_BASE_PATH}/src.zip submissions/.
	@if [ `stat --printf="%s" submissions/src.zip` -gt 21474836480 ]; \
	then \
		echo 'THE SUBMISSION IS TOO BIG. IT SHOULD BE LESS THAN 20GB.'; \
	fi
	mv submissions/src.zip submissions/$$(date +%s%N | cut -b1-13).zip
