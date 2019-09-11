###### START CHANGABLE ######

# path to the code
BASE_PATH     := code
# path to the data
DATA_PATH     := ./data/check

###### END CHANGABLE ######

SHELL         := /bin/bash

ABS_BASE_PATH := $$(realpath .)/${BASE_PATH}
IMAGE         := $$(jq -r ".image" ${BASE_PATH}/metadata.json)
RUNNER        := $$(jq -r ".entry_point" ${BASE_PATH}/metadata.json)

all:
	@echo "Please specify target"

predict: run test destroy
	@echo "$@ is done!"

run:
	sudo docker run \
		-d \
		-v ${ABS_BASE_PATH}:/root/code \
		--net="host" \
		--name="tester" \
		${IMAGE} \
		/bin/bash -c "cd /root/code && ${RUNNER}"

test:
	cd client && \
	python sender.py --folder-path ../${DATA_PATH} --url http://localhost:8000 && \
	cd ..

destroy:
	sudo docker stop tester
	sudo docker rm tester

submit:
	pushd ${ABS_BASE_PATH} && \
	zip -r code.zip * && \
	popd && \
	cp ${ABS_BASE_PATH}/code.zip submissions/.
