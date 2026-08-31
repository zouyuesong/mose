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
	$(GOBUILD) -o ./bin/csvport -v ./main/main.go

.PHONY: clean
clean:
	$(GOCLEAN)
	rm -f ./bin/*