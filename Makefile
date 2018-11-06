TARGET=
.PHONY: all full_test test build

all: full_test

test:
	docker run -it -v $(shell pwd):/src tensorboardx:test pytest ${TARGET}

full_test:
	docker run -it -v $(shell pwd):/src tensorboardx:test

build:
	docker build -t tensorboardx:test .
