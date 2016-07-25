GIT_DL_LINK := https://github.com/tensorflow/tensorflow.git

default_target: all

all: tensorflow/bazel-bin

.PRECIOUS: tensorflow/bazel-bin
tensorflow/bazel-bin:
	$(MAKE) configure
	@cd tensorflow && bazel build -c opt //tensorflow/cc:tutorials_example/trainer

.PRECIOUS: configure
configure: tensorflow/configure
	@cd tensorflow && \
		TF_NEED_GCP=0 TF_NEED_CUDA=0 \
		PYTHON_BIN_PATH=`which python` \
		SWIG_PATH=/usr/bin/swig \
			./configure

.PRECIOUS: tensorflow/configure
tensorflow/configure:
	@echo "\nDownloading tensorflow \n\n"
	git clone $(GIT_DL_LINK)
	git apply ../patch.diff
	
clean:
	@cd tensorflow && bazel --clean expunge
