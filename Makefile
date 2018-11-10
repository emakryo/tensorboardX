TARGET=
.PHONY: all full_test test build clean

all: full_test

test:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test pytest ${TARGET}

full_test:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test

build:
	docker build -t tensorboardx:test .

tensorboard:
	docker run -v $(shell pwd):/src -p 6006:6006 --rm tensorflow/tensorflow:1.12.0-py3 tensorboard --logdir=/src/runs

clean:
	rm -rf runs
