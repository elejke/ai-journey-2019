###### START CHANGABLE ######

# path to the code
BASE_PATH     := .
# path to the data
DATA_PATH     := ./data/check

###### END CHANGABLE ######

ABS_BASE_PATH := $$(realpath .)/${BASE_PATH}
IMAGE         := $$(jq -r ".image" ${BASE_PATH}/metadata.json)
RUNNER        := $$(jq -r ".entry_point" ${BASE_PATH}/metadata.json)

TIMESTAMP     := $(shell date +%s%N | cut -b1-13)

all:
	@echo "Please specify target"

predict: create predictor destroy
	@echo "$@ is done!"

evaluate: create evaluator destroy
	@echo "$@ is done!"

create:
	sudo docker pull ${IMAGE}
	AVAILABLE_CORES=$$(cat /proc/cpuinfo | grep processor | wc -l); \
	DESIRABLE_CORES=4; \
	CPUS_LIMIT=$$((AVAILABLE_CORES > DESIRABLE_CORES ? DESIRABLE_CORES : AVAILABLE_CORES)); \
	sudo docker run \
		-d \
		-v ${ABS_BASE_PATH}/src:/root/solution/src \
		-v ${ABS_BASE_PATH}/models:/root/solution/models \
		-p 8000:8000 \
		--memory="16g" \
		--memory-swap="16g" \
		--cpus=$${CPUS_LIMIT} \
		--name="tester-${TIMESTAMP}" \
		${IMAGE} \
		/bin/bash -c "cd /root/solution && ${RUNNER}"

predictor:
	cd client && \
	python3 predictor.py --folder-path ../${DATA_PATH} --url http://localhost:8000 && \
	cd ..

evaluator:
	- sudo docker run \
		-v ${ABS_BASE_PATH}/client:/root/solution/client \
		-v ${ABS_BASE_PATH}/data:/root/solution/data \
		--rm \
		--net="host" \
		--name="querier-${TIMESTAMP}" \
		${IMAGE} \
		/bin/bash -c "cd /root/solution/client && python3 evaluator.py \
															--folder-path ../${DATA_PATH} \
															--url http://localhost:8000"

destroy:
	sudo docker stop tester-${TIMESTAMP}
	sudo docker rm tester-${TIMESTAMP}

destroy_all:
	sudo docker stop $$(sudo docker ps -aq --filter="name=tester-*")
	sudo docker rm $$(sudo docker ps -aq --filter="name=tester-*")

submit:
	cd ${ABS_BASE_PATH} && \
	zip -r src.zip src models/dictionaries models/task_16 metadata.json -x *__pycache__* && \
	cd - && \
	mv ${ABS_BASE_PATH}/src.zip submissions/.
	@if [ `stat --printf="%s" submissions/src.zip` -gt 21474836480 ]; \
	then \
		echo 'THE SUBMISSION IS TOO BIG. IT SHOULD BE LESS THAN 20GB.'; \
	fi
	mv submissions/src.zip submissions/$$(date +%s%N | cut -b1-13).zip

build:
	cp dockers/aij/.dockerignore .
	sudo docker build -f dockers/aij/Dockerfile -t aij .
	rm .dockerignore
