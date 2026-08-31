# Go parameters
GOCMD=go
GOLINT=golint
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
#BUILD TARGET

all: build

build: apps

.PHONY: apps
apps:
	$(GOBUILD) -o ./bin/jobd -v ./main

.PHONY: clean
clean:
	$(GOCLEAN)
	rm -f ./bin/*