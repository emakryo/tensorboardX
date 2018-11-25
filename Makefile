TARGET=
.PHONY: all full_test test build clean

all: test_full

test:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test pytest ${TARGET}

test_full: test_all_chainer test_all_no_chainer

test_all_chainer:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test

test_all_no_chainer:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test-no-chainer

flake8:
	docker run -it -v $(shell pwd):/src --rm tensorboardx:test flake8 tensorboardX

build: build_chainer build_no_chainer

build_chainer:
	docker build -t tensorboardx:test

build_no_chainer:
	docker build --no-cache --build-arg WITH_CHAINER=0 -t tensorboardx:test-no-chainer .

tensorboard:
	docker run -v $(shell pwd):/src -p 6006:6006 --rm tensorflow/tensorflow:1.12.0-py3 tensorboard --logdir=/src/runs

clean:
	rm -rf runs
