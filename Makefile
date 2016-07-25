dl_link := https://github.com/tensorflow/tensorflow/archive/
dl_file := v0.9.0.zip
unzip_dir := tensorflow

default_target: all

all: $(unzip_dir)/bazel-bin

.PRECIOUS: $(unzip_dir)/bazel-bin
$(unzip_dir)/bazel-bin:
	$(MAKE) configure
	@cd $(unzip_dir) && \
		bazel build -c opt //tensorflow/cc:tutorials_example_trainer

.PRECIOUS: configure
configure: $(unzip_dir)/configure
	@cd $(unzip_dir) && \
		TF_NEED_GCP=0 TF_NEED_CUDA=0 \
		PYTHON_BIN_PATH=`which python` \
		SWIG_PATH=/usr/bin/swig \
			./configure

.PRECIOUS: $(unzip_dir)/configure
$(unzip_dir)/configure:
	@echo "\nDownloading tensorflow \n\n"
	wget -T 60 $(dl_link)/$(dl_file) -O $(dl_file)
	@echo "\nUnzipping to $(unzip_dir) \n\n"
	unzip $(dl_file) && rm $(dl_file) && mv tensorflow-0.9.0 $(unzip_dir)

clean:
	@cd $(unzip_dir) && bazel --clean expunge
